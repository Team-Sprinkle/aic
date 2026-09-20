"""Process-isolated causal inference; CPU arrays cross a private local pipe.

The ROS process owns observations, physical bounds, and publication. Only the
unchanged AICController and its causal caches live in the child process.
"""
from __future__ import annotations
import atexit,os,pickle,select,struct,subprocess,sys,time
from pathlib import Path


def _write(stream,value):
    payload=pickle.dumps(value,protocol=5)
    pending=memoryview(struct.pack('!Q',len(payload))+payload)
    while pending:
        written=os.write(stream.fileno(),pending)
        if written<=0:raise EOFError('Dreamer inference pipe write failed')
        pending=pending[written:]


def _read(stream,timeout=None):
    deadline=None if timeout is None else time.monotonic()+timeout
    def exact(size):
        chunks=[]
        while size:
            remaining=None if deadline is None else max(0.,deadline-time.monotonic())
            if not select.select([stream],[],[],remaining)[0]:
                raise TimeoutError('Dreamer inference worker response timed out')
            chunk=os.read(stream.fileno(),size)
            if not chunk:raise EOFError('Dreamer inference pipe closed')
            chunks.append(chunk);size-=len(chunk)
        return b''.join(chunks)
    size=struct.unpack('!Q',exact(8))[0]
    if size>64*1024*1024:raise ValueError('Unexpected inference message size')
    return pickle.loads(exact(size))


class RemoteAICController:
    def __init__(self,checkpoint,source,device,precision='bf16'):
        self.offset=0
        self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(checkpoint),str(source),str(device),precision],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        atexit.register(self.close)
        ready=_read(self.process.stdout,90.)
        if ready.get('status')!='ready':raise RuntimeError(ready)
        self.worker_metadata=ready
    def _request(self,value):
        if self.process.poll() is not None:raise RuntimeError('Dreamer worker exited before inference')
        _write(self.process.stdin,value);answer=_read(self.process.stdout,15.)
        if answer.get('status')!='ok':raise RuntimeError(answer)
        self.offset=int(answer['offset']);return answer
    def reset(self):self._request({'command':'reset'})
    def act(self,images,states,task,time_sec,executed_previous=None):
        return self._request({'command':'act','args':(images,states,task,time_sec,executed_previous)})['action']
    def close(self):
        process=getattr(self,'process',None)
        if process is None:return
        if process.poll() is None:
            try:process.stdin.close();process.wait(timeout=3.)
            except (OSError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=2.)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def worker(checkpoint,source,device,precision):
    import ctypes,signal,traceback
    original_parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=original_parent:return
    sys.path.insert(0,source)
    import torch
    from dreamer4.aic.train import load
    from dreamer4.aic.models import AICController
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model,saved=load(checkpoint,device,control_only=True)
    controller=AICController(model,saved['normalization'],device,precision)
    device=torch.device(device)
    metadata={'status':'ready','pid':os.getpid(),'device':str(device),'uuid':str(torch.cuda.get_device_properties(device).uuid) if device.type=='cuda' else None,'torch_intraop':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads()}
    _write(sys.stdout.buffer,metadata)
    while True:
        try:request=_read(sys.stdin.buffer)
        except EOFError:return
        try:
            start=time.perf_counter()
            if request['command']=='reset':controller.reset();action=None
            elif request['command']=='act':action=controller.act(*request['args'])
            else:raise ValueError('Unknown worker command')
            _write(sys.stdout.buffer,{'status':'ok','offset':controller.offset,'action':action,'worker_ms':(time.perf_counter()-start)*1000})
        except Exception:
            _write(sys.stdout.buffer,{'status':'error','traceback':traceback.format_exc()});return


if __name__=='__main__':
    assert len(sys.argv)==6 and sys.argv[1]=='--worker'
    worker(*sys.argv[2:])
