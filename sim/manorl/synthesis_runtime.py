"""Reusable MJX-Warp rollout runtime for the fivefold raw120 campaign."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.synthesis_campaign import CAMPAIGN_CONTRACT, canonical_hash
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch


class BatchSynthesisRuntime:
    """One object/scene topology, one checkpoint, one fixed vector width.

    The Warp model and policy persist across waves. Only references, action IDs,
    reset positions and seeded point templates change. Compact collection never
    retains observations, policy actions or PPO transition arrays.
    """
    def __init__(self, trajectories: list[ReferenceTrajectory], checkpoint: Path, *,
                 ccd_contacts: int = 16, contacts_per_world: int = 1024,
                 constraint_capacity: int = 4096, reuse_contact_snapshot: bool = True):
        from sim.manorl import assets
        from sim.manorl.checkpoint import checkpoint_runtime_metadata
        from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
        from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
        from sim.manorl.rewards import RewardConfig
        from sim.manorl.view_environment import _checkpoint_environment_options, _build_checkpoint_stepper
        from sim.manorl.lance_v2 import file_sha256
        self.checkpoint = checkpoint
        self.metadata = checkpoint_runtime_metadata(checkpoint)
        self.options = _checkpoint_environment_options(checkpoint)
        saved = self.metadata['runtime_config']['environment']
        self.saved = saved
        self.object_name = trajectories[0].identity.identity.rsplit('_', 2)[0]
        self.scene_names = tuple(trajectories[0].scene_object_types or (self.object_name,))
        self.batch_size = len(trajectories)
        self._check_topology(trajectories)
        for name, actual in assets.asset_provenance().items():
            if saved.get(name) != actual:
                raise ValueError(f'checkpoint physical asset mismatch: {name}')
        if self.options.pre_padding != 80 or self.options.reference_fps != 120 or self.options.control_fps != 120:
            raise ValueError('campaign checkpoint must be pre80 at 120/480Hz')
        compatibility = replace(SOURCE_ALIGNED_COMPATIBILITY,
            movement_pre_padding=self.options.pre_padding,
            early_phase_steps=saved['compatibility']['early_phase_steps'],
            point_template_mode=saved['compatibility']['point_template_mode'])
        reward = RewardConfig(**saved['reward'])
        config = EnvironmentConfig(
            device='gpu', num_envs=self.batch_size, hand_side='right',
            residual_enabled=saved['residual_enabled'], residual_action=self.options.residual_action,
            compatibility=compatibility, reward_config=reward,
            max_deviation_distance=saved['max_deviation_distance'],
            reference_fps=120, control_fps=120, post_padding=self.options.post_padding,
            expected_contact_mode=self.options.expected_contact_mode,
            residual_joint_mode=saved['residual_joint_mode'],
            contact_capacity=contacts_per_world*self.batch_size,
            constraint_capacity=constraint_capacity,
            warp_ccd_iterations=self.options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=ccd_contacts,
            device_resident_controls=True, capture_transition_diagnostics=False)
        self.environment = MujocoManoEnvironment(TrajectoryBatch(tuple(trajectories)), config)
        env = self.environment
        env.producer.retain_materialized_snapshot = reuse_contact_snapshot
        self.stepper = _build_checkpoint_stepper(env, checkpoint, policy_transfer=True,
                                                policy_enable_steps=env.early_phase_lengths)
        # The transfer loader allows intentional prefix/capacity differences.
        # Verify the trained observation/control/physical fields explicitly.
        actual = self.stepper._runtime.agent.manorl_environment_signature
        fields = ('asset_manifest_sha256', 'asset_source_commit', 'action_dim', 'observation_dim',
                  'available_hand_sides', 'controlled_hand_sides', 'reference_fps', 'control_fps',
                  'physics_fps', 'expected_contact_mode', 'residual_joint_mode', 'action_penalty_scale')
        for name in fields:
            if json.dumps(actual.get(name), sort_keys=True) != json.dumps(saved.get(name), sort_keys=True):
                raise ValueError(f'synthesis changed trained semantics: {name}')
        self.checkpoint_sha = file_sha256(checkpoint)
        self.physical_manifest = json.loads(assets.ASSET_MANIFEST.read_text())
        self.object_bodies = {}
        self.object_geoms = {}
        self.object_qpos = {}
        for name in self.scene_names:
            runtime = assets.object_runtime(name)
            body = int(env.mujoco.mj_name2id(env.model, env.mujoco.mjtObj.mjOBJ_BODY, runtime.body_name))
            joint = int(env.mujoco.mj_name2id(env.model, env.mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name))
            if body < 0 or joint < 0:
                raise ValueError('scene object absent from compiled model')
            self.object_bodies[name] = body
            self.object_qpos[name] = int(env.model.jnt_qposadr[joint])
            self.object_geoms[name] = tuple(g for g in range(env.model.ngeom)
                if int(env.model.geom_bodyid[g]) == body and bool(env.model.geom_contype[g]))
        self.hand_geoms = set(env.producer.keypoint_geom_ids_by_side['right'])
        self.floor = int(env.mujoco.mj_name2id(env.model, env.mujoco.mjtObj.mjOBJ_GEOM, 'floor'))
        self.capacity_evidence = {}

    def _check_topology(self, trajectories):
        if len(trajectories) != self.batch_size:
            raise ValueError('runtime vector width changed')
        for t in trajectories:
            if (t.identity.identity.rsplit('_', 2)[0] != self.object_name
                or tuple(t.scene_object_types or (self.object_name,)) != self.scene_names
                or t.hand_sides != ('right',) or t.control_fps != 120):
                raise ValueError('runtime object/scene/hand/clock topology changed')

    def install(self, trajectories: list[ReferenceTrajectory], seeds: list[int]) -> None:
        from sim.manorl.environment import _build_masked_reset_data_fn, _expected_keypoint_ids
        self._check_topology(trajectories)
        env = self.environment
        env.trajectory = trajectories[0]; env.trajectories = tuple(trajectories)
        env.action_ids = np.array([int(t.identity.identity.rsplit('_',2)[1]) for t in trajectories], dtype=np.int64)
        env.expected_keypoint_ids = tuple(_expected_keypoint_ids(self.object_name, f'{action:02d}', env.config.expected_contact_mode)
                                         for action in env.action_ids)
        env.expected_contact_mask.fill(0)
        for i, ids in enumerate(env.expected_keypoint_ids):env.expected_contact_mask[i, list(ids)] = 1
        env.expected_contact_weights = env.expected_contact_mask.copy()
        if env.config.residual_joint_mode != 'all':
            raise ValueError('raw120 campaign currently requires trained all-joint residual authority')
        env._build_reference_tables()
        env._reset_qpos = env._initial_qpos()
        env._reset_qpos_device = env.jax.device_put(env._reset_qpos, env.device)
        env._reset_ctrl_device = env.jax.device_put(env.reference_q_model[:,0], env.device)
        env._reset_data_fn = _build_masked_reset_data_fn(jax=env.jax, jp=env.jp,
            reset_qpos=env._reset_qpos_device, reset_ctrl=env._reset_ctrl_device)
        env.reseed_point_templates_per_env(np.asarray(seeds, dtype=np.int64))
        observations, _ = self.stepper._runtime.env.reset()
        self.stepper._observations = observations
        self.stepper._pending_done = None
        self.stepper._policy_enable_steps = env.early_phase_lengths.copy()
        self.stepper._policy_disable_steps = None
        self.capacity_evidence = {'max_contacts':0,'max_constraints':0,
            'ccd_counter_available':bool(env._warp_ccd_overflow_guard_available),
            'ccd_contacts_per_world':env.config.warp_ccd_contacts_per_world,
            'contact_capacity':env.config.contact_capacity,'constraint_capacity':env.config.constraint_capacity}

    def _contacts(self):
        from sim.manorl.lance_v2 import corrected_contact_frames
        env = self.environment
        cached = getattr(env.producer, 'materialized_snapshot', None)
        if cached is not None and cached[0] is env.data and cached[2] is not None:
            _, state, buffers = cached
        else:
            state = env.producer.materialize_state(env.data)
            buffers = env.producer.materialize_contact_buffers(env.data, self.batch_size)
        contacts = [[] for _ in range(self.batch_size)]
        any_object = np.zeros(self.batch_size, dtype=bool)
        for name in self.scene_names:
            decoded = corrected_contact_frames(buffers=buffers,state=state,model=env.model,
                keypoint_geom_ids=env.producer.keypoint_geom_ids_by_side['right'],
                object_geom_ids=self.object_geoms[name],object_body_id=self.object_bodies[name],
                wrist_body_id=env.producer.keypoint_body_ids[0],object_name=name)
            for i, entries in enumerate(decoded):
                contacts[i].extend(entries)
                any_object[i] |= any(np.linalg.norm(pair['force_normal']) > .2 for entry in entries for pair in entry['contact_pairs'])
        floor = np.zeros(self.batch_size, dtype=bool)
        for i in range(buffers.count):
            a,b=map(int,buffers.geom[i])
            if (a==self.floor and b in self.hand_geoms) or (b==self.floor and a in self.hand_geoms):
                w=int(buffers.world[i]); address=np.asarray(buffers.addresses[i],dtype=int)
                valid=address[(address>=0)&(address<buffers.nefc[w])]
                floor[w] |= float(np.sum(buffers.constraint_force[w,valid])) > .2
        self.capacity_evidence['max_contacts']=max(self.capacity_evidence['max_contacts'],buffers.count)
        self.capacity_evidence['max_constraints']=max(self.capacity_evidence['max_constraints'],int(np.max(buffers.nefc)))
        return contacts, floor, any_object, state

    def collect(self, trajectories: list[ReferenceTrajectory], samples: list[dict], claims: list[dict], *,
                progress: Callable[[dict], None] = lambda value: None) -> tuple[list[dict], dict]:
        from tools.export_manorl_synthetic_lance import _evaluate_collected_candidate
        env=self.environment; n=len(trajectories)
        stores=[defaultdict(list) for _ in range(n)]; commands=[defaultdict(list) for _ in range(n)]
        contact_frames=[[] for _ in range(n)]
        active=np.ones(n,dtype=bool); reasons=np.zeros(n,dtype=int); complete=np.zeros(n,dtype=bool)
        failures=[None]*n
        started=time.perf_counter(); phase={'physics_policy_seconds':0.,'contacts_seconds':0.,'collection_seconds':0.}
        t0=time.perf_counter(); contacts,floor,obj,state=self._contacts();phase['contacts_seconds']+=time.perf_counter()-t0
        def append(i, native, ctrl):
            p=env.last_physical
            stores[i]['urdf_dof'].append(np.array(p.mano_dof_pos[i],copy=True))
            stores[i]['urdf_dof_target'].append(np.array(ctrl[i],copy=True))
            stores[i]['mano_joint_pos'].append(np.concatenate([p.hand_keypoint_positions[i],p.fingertip_positions[i]]))
            stores[i]['object_position'].append(np.array(p.object_position[i],copy=True))
            stores[i]['object_orientation_xyzw'].append(np.array(p.object_orientation_xyzw[i],copy=True))
            stores[i]['scene_position'].append(np.array([native.xpos[i,self.object_bodies[name]] for name in self.scene_names]))
            stores[i]['scene_quaternion'].append(np.array([native.xquat[i,self.object_bodies[name]][[1,2,3,0]] for name in self.scene_names]))
        for i in range(n):
            append(i,state,env.reference_q_model[:,0]);contact_frames[i].append(contacts[i])
            if floor[i] or obj[i]:failures[i]='prefix_initial_collision';active[i]=False
        steps=0
        for step in range(int(env.trajectory_lengths.max())):
            if not active.any():break
            command_index=env._target_indices().copy()
            t0=time.perf_counter();self.stepper.step();phase['physics_policy_seconds']+=time.perf_counter()-t0
            t0=time.perf_counter();contacts,floor,obj,state=self._contacts();phase['contacts_seconds']+=time.perf_counter()-t0
            t0=time.perf_counter();ctrl=np.asarray(env.data.ctrl,dtype=np.float64)
            for i in np.flatnonzero(active):
                i=int(i);append(i,state,ctrl);contact_frames[i].append(contacts[i])
                commands[i]['command_reference_index'].append(int(command_index[i]))
                commands[i]['command_source_frame_index'].append(int(env.reference_source_indices[i,command_index[i]]))
                commands[i]['reward'].append(float(env.last_reward.total[i]))
                if command_index[i] < trajectories[i].augmentation_prefix_frames and (floor[i] or obj[i]):
                    failures[i]='prefix_collision';active[i]=False
                if env.last_termination.reset[i]:
                    reasons[i]=int(env.last_termination.reason_code[i]);complete[i]=bool(env.last_termination.success[i]);active[i]=False
            steps=step+1;phase['collection_seconds']+=time.perf_counter()-t0
            if steps%100==0:progress({'step':steps,'active':int(active.sum()),'worlds':n})
        if active.any():raise RuntimeError('synthesis exceeded complete reference horizon')
        results=[]
        for i,claim in enumerate(claims):
            gate=_evaluate_collected_candidate(trajectory=trajectories[i],state_storage=stores[i],
                trajectory_complete=bool(complete[i]),termination_reason_code=int(reasons[i]),contact_frames=contact_frames[i])
            diagnostic={'accepted':bool(gate.accepted and failures[i] is None),'gate':gate.to_dict(),
                        'prefix_failure':failures[i],'prefix':samples[i], 'capacity':dict(self.capacity_evidence)}
            results.append({'claim':claim,'diagnostic':diagnostic,'trajectory':trajectories[i],
                            'states':dict(stores[i]) if diagnostic['accepted'] else None,
                            'commands':dict(commands[i]) if diagnostic['accepted'] else None,
                            'contacts':contact_frames[i] if diagnostic['accepted'] else None})
        phase.update({'wall_seconds':time.perf_counter()-started,'steps':steps,'worlds':n,
                      'base_accepted':sum(r['diagnostic']['accepted'] for r in results),
                      'capacity':dict(self.capacity_evidence)})
        return results,phase

    def compact_with_retreat(self, result: dict, *, software_commit: str, collision_check: bool = True):
        from sim.manorl.lance_v2 import build_v2_row, build_compact_row
        from tools.build_manorl_full_retreat import last_contact_frame, build_full_retreat_row, ANCHOR_OFFSET
        if not result['diagnostic']['accepted']:return None
        trajectory=result['trajectory'];sample=result['diagnostic']['prefix'];states=result['states'];env=self.environment
        augmentation=CAMPAIGN_CONTRACT+':'+canonical_hash(sample)
        metadata={'gesture':trajectory.identity.identity.rsplit('_',2)[1], 'hand_names':['right'],
                  'mano_hand_shapes':[self.physical_manifest['hands']['right']['betas']]}
        provenance={'checkpoint_path':str(self.checkpoint), 'checkpoint_sha256':self.checkpoint_sha,
            'checkpoint_update':self.metadata['runtime_config'].get('training_progress',{}).get('completed_updates',-1),
            'checkpoint_metadata':self.metadata, 'software_commit':software_commit, 'seed':sample['seed'],
            'episode_index':result['claim']['slot'],'generation_attempt':result['claim']['attempt'],
            'augmentation_identity':augmentation}
        full=build_v2_row(trajectory=trajectory,source_index={'operator':self.physical_manifest['hand_operator'],
            'capMachine':'manorl-mjx-warp'},source_metadata=metadata,
            states={k:np.asarray(v) for k,v in states.items() if not k.startswith('scene_')},
            contacts=result['contacts'],rollout={k:np.asarray(v) for k,v in result['commands'].items()},
            provenance=provenance,include_checkpoint_metadata_json=False)
        row=build_compact_row(full,checkpoint_metadata=self.metadata,
            warp_ccd_iterations=env.config.warp_ccd_iterations,warp_ccd_contacts_per_world=env.config.warp_ccd_contacts_per_world)
        row['objects']=[{'pos':np.asarray(states['scene_position'])[:,j].astype(np.float32).tolist(),
                         'rot_aa':Rotation.from_quat(np.asarray(states['scene_quaternion'])[:,j]).as_rotvec().astype(np.float32).tolist()}
                        for j in range(len(self.scene_names))]
        row['trajectory_metadata']['object_names']=list(self.scene_names)
        contact_ends=[last_contact_frame(row,name) for name in self.scene_names]
        last=max((value for value in contact_ends if value is not None),default=None)
        if last is None:
            result['diagnostic'].update(accepted=False,retreat_failure='no_contact');return None
        try:
            updated,diag=build_full_retreat_row(row,anchor=last+ANCHOR_OFFSET,seed=sample['seed'],batch_tag='raw120-fivefold',
                joint_lower=env.joint_lower,joint_upper=env.joint_upper)
        except ValueError as exc:
            result['diagnostic'].update(accepted=False,retreat_failure=str(exc));return None
        if updated is None:
            result['diagnostic'].update(accepted=False,retreat_failure=diag);return None
        # A root translation also translates every world-space hand keypoint and
        # the kinematic tail's command. Nominal source reference remains unchanged.
        before=np.asarray(row['hands'][0]['urdf_dof'])
        after=np.asarray(updated['hands'][0]['urdf_dof'])
        delta=after[:,:3]-before[:,:3]
        updated['hands'][0]['mano_joint_pos']=(np.asarray(row['hands'][0]['mano_joint_pos'])+delta[:,None,:]).astype(np.float32).tolist()
        targets=np.asarray(row['hands'][0]['urdf_dof_target']).copy();targets[:,:3]+=delta
        if np.any(targets < env.joint_lower[None]-1e-5) or np.any(targets > env.joint_upper[None]+1e-5):
            result['diagnostic'].update(accepted=False,retreat_failure='kinematic_target_limits');return None
        updated['hands'][0]['urdf_dof_target']=targets.astype(np.float32).tolist()
        if collision_check:
            collision=self._retreat_collision(updated,diag['anchor'])
            if collision is not None:
                result['diagnostic'].update(accepted=False,retreat_failure=collision);return None
        updated['provenance']['augmentation_identity']=augmentation+':offline_retreat_xyz_targets_keypoints_v1'
        result['diagnostic']['retreat']={**diag,'contract':'offline_retreat_xyz_targets_keypoints_v1',
            'collision_check':'native_kinematic_geometry_only' if collision_check else 'not_run',
            'contacts':'retained_original_warp_solution_no_new_force_solve'}
        return updated

    def _retreat_collision(self,row,anchor):
        env=self.environment;mj=env.mujoco;data=mj.MjData(env.model)
        q=np.asarray(row['hands'][0]['urdf_dof'])
        object_rot=[Rotation.from_rotvec(np.asarray(o['rot_aa'])).as_quat() for o in row['objects']]
        for t in range(anchor+1,len(q)):
            data.qpos[:28]=q[t]
            for j,name in enumerate(self.scene_names):
                address=self.object_qpos[name];data.qpos[address:address+3]=row['objects'][j]['pos'][t]
                data.qpos[address+3:address+7]=object_rot[j][t][[3,0,1,2]]
            mj.mj_kinematics(env.model,data);mj.mj_comPos(env.model,data);mj.mj_collision(env.model,data)
            for c in data.contact[:data.ncon]:
                a,b=int(c.geom1),int(c.geom2)
                if ((a in self.hand_geoms) != (b in self.hand_geoms)) and float(c.dist)<-1e-5:
                    return {'reason':'offline_retreat_geometry_penetration','frame':t,'depth_m':float(c.dist)}
        return None
