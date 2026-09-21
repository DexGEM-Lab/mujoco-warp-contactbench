#!/usr/bin/env python3
"""Live native U1 grasp/carry/release workbench; no state-forcing controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import time
import numpy as np
from tools.u1_interactive_session import Session, parse_command, validate_restore


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


class Workspace:
    def __init__(self, args, session):
        import tkinter as tk
        from tkinter import ttk
        import mujoco as mj
        import mujoco.viewer
        self.args,self.s,self.mj=args,session,mj
        self.output=args.output
        self.inbox=self.output/'commands';self.inbox.mkdir()
        (self.output/'command_history').mkdir()
        self.queue=Queue();self.running=False;self.budget=args.checkpoint
        self.auto_save=True;self.saved=None;self.last_draw=0.;self.last_step=0.
        self.root=tk.Tk();self.root.title(f'{args.label} | native U1 | grasp → carry → release')
        self.root.geometry('850x950+10+30')
        self.status=tk.StringVar(value='Frame0 physical replay to checkpoint…')
        ttk.Label(self.root,textvariable=self.status,wraplength=820).pack(fill='x')
        row=ttk.Frame(self.root);row.pack(fill='x')
        for label,m in [('Play/pause',{'action':'play'}),('Step1',{'action':'step','frames':1}),('Step12',{'action':'step','frames':12}),('Save',{'action':'save'}),('Restore',{'action':'restore'}),('Reset/clear',{'action':'reset'}),('Export',{'action':'export'})]:
            ttk.Button(row,text=label,command=lambda cmd=m:self.safe_command(cmd)).pack(side='left')
        row=ttk.Frame(self.root);row.pack(fill='x')
        self.goto=tk.IntVar(value=args.checkpoint)
        ttk.Entry(row,textvariable=self.goto,width=7).pack(side='left')
        ttk.Button(row,text='Replay frame0 → checkpoint (keep edits)',command=lambda:self.safe_command({'action':'goto','frame':self.goto.get()})).pack(side='left')
        self.start=tk.IntVar(value=args.checkpoint);self.end=tk.IntVar(value=min(args.checkpoint+120,len(session.target)-1));self.ramp=tk.IntVar(value=24)
        row=ttk.Frame(self.root);row.pack(fill='x')
        for label,var in [('Start',self.start),('End',self.end),('Ramp frames',self.ramp)]:
            ttk.Label(row,text=label).pack(side='left');ttk.Entry(row,textvariable=var,width=7).pack(side='left')
        ttk.Label(self.root,text='Add offsets in metres (XYZ) / radians (Euler, fingers). Smooth in/out; zero at window endpoints.').pack()
        book=ttk.Notebook(self.root);book.pack(fill='x')
        self.values=[]
        for title,indices in [('Wrist XYZ/Euler',range(6)),('Thumb',range(6,12)),('Index/middle',range(12,20)),('Ring/pinky',range(20,28))]:
            pane=ttk.Frame(book);book.add(pane,text=title)
            for j in indices:
                row=ttk.Frame(pane);row.pack(fill='x')
                ttk.Label(row,text=f'{j}: {session.model.joint(j).name}',width=38).pack(side='left')
                v=tk.DoubleVar(value=0);self.values.append(v)
                ttk.Entry(row,textvariable=v,width=10).pack(side='left')
                ttk.Button(row,text='Add window',command=lambda index=j,var=v:self.safe_command(dict(action='offset',joint=index,value=var.get(),start=self.start.get(),end=self.end.get(),ramp=self.ramp.get()))).pack(side='left')
        self.text=tk.Text(self.root,height=24,width=110);self.text.pack(fill='both',expand=True)
        ttk.Label(self.root,text='Viewer keys: Space play/pause | N step1 | K save | R restore | E export. Contacts: red spheres.').pack()
        self.render=mj.MjData(session.model);self.render.qpos[:]=session.data.qpos.numpy()[0]
        mj.mj_forward(session.model,self.render)
        self.viewer=mujoco.viewer.launch_passive(session.model,self.render,key_callback=self.key,show_left_ui=False,show_right_ui=False)
        self.viewer.cam.lookat[:]=session.cpu.xpos[session.object_body]
        self.viewer.cam.distance=1.3;self.viewer.cam.azimuth=120;self.viewer.cam.elevation=-25
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(10,self.tick)

    def key(self,key):
        commands={32:dict(action='play'),78:dict(action='step',frames=1),75:dict(action='save'),82:dict(action='restore'),69:dict(action='export')}
        if key in commands:self.queue.put(commands[key])

    def safe_command(self,m):
        try:return self.command(m)
        except Exception as exc:
            self.running=False;self.budget=0;self.status.set(str(exc))
            return dict(accepted=False,error=str(exc))

    def command(self,message):
        m=parse_command(message,len(self.s.target));a=m['action'];s=self.s
        if a=='play':self.running=not self.running;self.budget=0
        elif a=='pause':self.running=False;self.budget=0
        elif a=='step':self.running=False;self.budget=m['frames']
        elif a=='save':
            self.running=False;self.budget=0
            self.saved=s.save_checkpoint(self.output/'checkpoints'/stamp())
        elif a=='restore':
            if self.saved is None:raise ValueError('no saved checkpoint')
            s.restore(self.saved);self.running=False;self.budget=0
        elif a in ('reset','goto'):
            s.reset(clear=a=='reset');self.running=False;self.budget=m.get('frame',0);self.auto_save=a=='goto'
        elif a=='offset':s.offset(m)
        elif a=='export':s.export(self.output/'candidates'/stamp())
        return dict(accepted=True,action=a,frame=s.frame,budget=self.budget)

    def draw(self):
        s=self.s;state=s.state();mj=self.mj
        with self.viewer.lock():
            self.render.qpos[:]=s.cpu.qpos;self.render.qvel[:]=s.cpu.qvel
            mj.mj_forward(s.model,self.render)
            scene=self.viewer.user_scn;scene.ngeom=0
            for c in state['contacts']:
                if scene.ngeom>=scene.maxgeom:break
                mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_SPHERE,np.full(3,.003),np.array(c['position']),np.eye(3).ravel(),np.array([1.,.1,.1,1.]))
                scene.ngeom+=1
        self.viewer.sync(state_only=True)
        self.status.set(f"U1 frame {s.frame}/{len(s.target)-1} | {'PLAY' if self.running else 'PAUSED' if not self.budget else 'STEPPING'} | links {state['hand_links']} | tilt {state['object_tilt_deg']:.2f}°")
        shown={k:state[k] for k in ('contact_count','ray_count','object_position','object_quaternion_wxyz','hand_object_transform','teacher_hand_object_transform','teacher_hand_links','contacts','teacher_contacts','current_28d','desired_28d')}
        self.text.delete('1.0','end');self.text.insert('end',json.dumps(shown,indent=1))
        state.update(running=self.running,remaining_steps=self.budget,checkpoint_frame=None if self.saved is None else self.saved['frame'],pid=os.getpid())
        p=self.output/'state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)

    def tick(self):
        if not self.viewer.is_running():self.close();return
        try:
            while not self.queue.empty():self.safe_command(self.queue.get_nowait())
            for p in sorted(self.inbox.glob('*.json')):
                try:result=self.safe_command(json.loads(p.read_text()))
                except Exception as exc:result=dict(accepted=False,error=str(exc))
                history=self.output/'command_history'
                p.rename(history/p.name);(history/(p.stem+'.response.json')).write_text(json.dumps(result,indent=2))
            if self.budget:
                for _ in range(min(self.budget,4)):
                    self.budget-=1
                    if not self.s.step():self.budget=0;break
            elif self.running and time.monotonic()-self.last_step>1/30:
                if not self.s.step():self.running=False
                self.last_step=time.monotonic()
            if self.auto_save and not self.budget:
                self.auto_save=False
                print('RESTORE_CHECK',validate_restore(self.s,self.output),flush=True)
                self.saved=self.s.save_checkpoint(self.output/'checkpoints'/stamp())
                print('CHECKPOINT_READY',self.s.frame,flush=True)
            if time.monotonic()-self.last_draw>.15:self.draw();self.last_draw=time.monotonic()
        except Exception:
            import traceback
            self.running=False;self.budget=0;self.auto_save=False
            text=traceback.format_exc();print(text,flush=True);(self.output/'error.txt').write_text(text);self.status.set(text)
        self.root.after(10,self.tick)

    def close(self):
        self.viewer.close();self.root.destroy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('bundle','target','dataset','asset-manifest'):p.add_argument('--'+name,type=Path)
    p.add_argument('--version',type=int)
    p.add_argument('--row',type=int)
    p.add_argument('--checkpoint',type=int,default=1030)
    a=p.parse_args()
    if a.dataset is not None:
        if a.version is None or a.row is None or a.asset_manifest is None or a.bundle:
            p.error('raw mode requires --version --row --asset-manifest, excludes --bundle, and permits an optional --target')
        from tools.u1_raw_source import load_raw_source
        inp,model,source_target,teacher,provenance=load_raw_source(a.dataset,a.version,a.row,a.asset_root,a.asset_manifest)
        target=source_target if a.target is None else np.load(a.target,allow_pickle=False)
        if target.shape != source_target.shape or not np.isfinite(target).all():
            p.error('raw --target must be a finite array matching the source [frames,28] shape')
        provenance=dict(provenance,target_role=('recorded_right_qpos_grounded' if a.target is None else 'explicit_U1_direct_target'),
                        target_path=None if a.target is None else str(a.target.resolve()))
        a.label=f'raw v{a.version} row{a.row}'
    else:
        if not a.bundle or not a.target or a.version is not None or a.row is not None or a.asset_manifest:
            p.error('historical mode requires --bundle --target and excludes raw arguments')
        from tools.pilot_start_augmentation import reconstruct
        from tools.run_uniform_direct_parent_canary import compile_model
        inp,_,_,teacher,provenance=reconstruct(a.bundle,'A_row035')
        _,model=compile_model(a.bundle,a.asset_root,inp,'U1')
        target=np.load(a.target,allow_pickle=False)
        a.label='A_row035'
    parse_command(dict(action='goto',frame=a.checkpoint),len(target))
    a.output.mkdir(parents=True,exist_ok=False)
    s=Session(inp,model,target,teacher,provenance)
    integer_checkpoint=parse_command(dict(action='goto',frame=a.checkpoint),len(s.target))
    (a.output/'source.json').write_text(json.dumps(dict(target=str(a.target.resolve()) if a.target else 'recorded_right_qpos_grounded',source=provenance,contract=s.contract),indent=2))
    w=Workspace(a,s)
    print('WORKSPACE_READY DISPLAY',os.environ.get('DISPLAY'),flush=True)
    w.root.mainloop()


if __name__=='__main__':main()
