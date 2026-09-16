// Exact index/phase formulas from the current kernels; this is not a GPU timing simulator.
function lessonConfig(version,name,sms=148){
 const G=version>=8?2:1,C=version===9?2:1,S=version<=4?1:version<=6?2:version===8?6:4;
 const BM=128*G,BN=128*G,mt=version<=2?1:4096/(BM*C),nt=version<=2?1:4096/BN;
 return {version,name,G,C,S,BM,BN,mt,nt,total:mt*nt,workers:version>=6?Math.min(mt*nt,Math.floor(sms/G)):mt*nt};
}
function grouped(ti,mt,nt){const group=Math.floor(ti/(8*nt)),base=group*8,rows=Math.min(8,mt-base),local=ti%(8*nt);return [base+local%rows,Math.floor(local/rows)];}
function lessonState(cfg,nk,kt,ti,c=0,peer=0,ei=0){
 const {version:v,G,C,S,BM,BN,mt,nt,workers}=cfg;
 const [bm,bn]=v>=6?grouped(ti,mt,nt):[ti%mt,Math.floor(ti/mt)];
 const worker=v>=6?ti%workers:ti,round=v>=6?Math.floor(ti/workers):0;
 const iteration=round*nk+kt,stage=v>=7?iteration%S:kt%S;
 const stageUses=Math.floor((nk+S-1-stage)/S);
 const full=v<=3?null:v<=6?(round*stageUses+Math.floor(kt/S))%2:Math.floor(iteration/S)%2;
 const empty=v>=7&&iteration>=S?(Math.floor(iteration/S)-1)%2:null;
 return {bm,bn,worker,round,iteration,stage,full,empty,done:iteration%2,
 rowBase:(bm*C+c)*BM+peer*128,colBase:bn*BN,
 inputBBase:bn*BN+peer*128,tmemBase:c*BN,epicol:bn*BN+ei*64,
 accFull:round%2,accEmpty:round>0?(round-1)%2:null};
}
const cfg=lessonConfig(v,data.name,data.result?.sms??148);
function bounded(id,max){let z=Number($(id).value);z=Number.isFinite(z)?Math.floor(z):0;z=Math.max(0,Math.min(max,z));$(id).value=z;return z;}
function draw(){
 const nk=bounded('nk',16)||1;$('nk').value=nk;$('kt').max=nk-1;
 const k=bounded('kt',nk-1),ti=bounded('tile',cfg.total-1),c=bounded('consumer',cfg.C-1),peer=bounded('peer',cfg.G-1),ei=bounded('ei',cfg.BN/64-1);
 const x=lessonState(cfg,nk,k,ti,c,peer,ei),S=cfg.S;
 $('round').max=v>=6?Math.floor((cfg.total-1-x.worker)/cfg.workers):0;$('round').value=x.round;
 $('ktLabel').textContent=`kt=${k} / nk=${nk}，K=${64*nk}`;
 $('buffers').replaceChildren();for(let st=0;st<S;st++){let e=document.createElement('div');e.className='buffer'+(st===x.stage?' current':'');e.textContent='stage '+st+(st===x.stage?' ← 选中批次':'');$('buffers').append(e);}
 let phase=v<=3?`MMA 完成 barrier phase=${x.done}`:v<=6?`full[${x.stage}] phase=${x.full}；done phase=${x.done}`:`full[${x.stage}] phase=${x.full}；${x.empty===null?'首次使用该输入槽，无须等旧 empty':'复用前等 empty phase='+x.empty}\nacc_full[c] phase=${x.accFull}；${x.accEmpty===null?'首次输出，无须等旧 acc_empty':'开始新输出前等 acc_empty phase='+x.accEmpty}`;
 $('mapping').textContent=`任务 ti=${ti} → (bm,bn)=(${x.bm},${x.bn})\n${v>=6?'worker='+x.worker+'，输出轮 round='+x.round+'；下一个任务 ti 加 '+cfg.workers:'每CTA一个输出任务；round=0'}\nK 输入 [${64*k},${64*(k+1)})；${v>=7?'连续 iteration='+x.iteration:'stage按本输出内 kt 选择'}\n${phase}\nconsumer=${c}，peer=${peer}：本侧 A 行 [${x.rowBase},${x.rowBase+128})；本侧 B 行 [${x.inputBBase},${x.inputBBase+128})\n本侧输出：行 [${x.rowBase},${x.rowBase+128}) × 列 [${x.colBase},${x.colBase+cfg.BN})\n本侧 TMEM slot 列 [${x.tmemBase},${x.tmemBase+cfg.BN})\n${v>=4?'ei='+ei+'：输出列 ['+x.epicol+','+(x.epicol+64)+')；局部写回tid3处理 D['+(x.rowBase+3)+','+(x.epicol+5)+']，CTA线程号 '+(128*c+3):'直接REG→GMEM；无输出TMA ei循环'}${data.name==='v09_tuned'?'\n本页 atom=32dp32b32x；每64列片按宽atom分成两组32列。':''}`;
 $('roles').replaceChildren();const roles=v<=3?['128线程：普通输入copy及结果写回','warp0：MMA提交']:v<=6?['warp0选一线程：TMA load/store','warp0：MMA提交','128线程：TMEM→REG→输出SMEM']:['WG0：consumer0写回',...(cfg.C===2?['WG1：consumer1写回']:[]),`warp${4*cfg.C+3}：本侧producer`,...Array.from({length:cfg.C},(_,c)=>`warp${4*cfg.C+c}：MMA c${c}`+(cfg.G===2?'（仅leader）':''))];
 for(const r of roles){let e=document.createElement('div');e.className='role';e.textContent=r;$('roles').append(e);}
 const ctx=$('tilemap').getContext('2d');ctx.clearRect(0,0,640,260);ctx.font='13px sans-serif';ctx.fillStyle='#334155';ctx.fillText('输出任务网格：蓝色共用B，绿色共用A，黄色当前任务；M向下，N向右',8,18);
 const w=570/cfg.nt,h=210/cfg.mt;for(let m=0;m<cfg.mt;m++)for(let n=0;n<cfg.nt;n++){ctx.fillStyle=m===x.bm&&n===x.bn?'#facc15':n===x.bn?'#93c5fd':m===x.bm?'#5eead4':'#e2e8f0';ctx.fillRect(35+n*w,35+m*h,w-2,h-2);}
 $('tmemmap').replaceChildren();for(let p=0;p<cfg.G;p++){let box=document.createElement('div');box.className='sm';box.textContent=`peer${p} / 本侧SM，TMEM 128 lanes`;for(let cc=0;cc<cfg.C;cc++){let rg=document.createElement('span');rg.className='range'+(cc===c?' chosen':'');rg.textContent=`c${cc} cols [${cc*cfg.BN},${(cc+1)*cfg.BN})`;box.append(rg);}$('tmemmap').append(box);}
 let rows='';const limit=Math.min(nk*(v>=6?2:1),24);
 for(let i=0;i<limit;i++){const rr=Math.floor(i/nk),kk=i%nk;const localTi=v>=6?x.worker+rr*cfg.workers:ti;if(localTi>=cfg.total)break;const y=lessonState(cfg,nk,kk,localTi,c,peer,ei);rows+=`<tr><td>${rr}</td><td>${kk}</td><td>${y.stage}</td><td>${y.full===null?'无 full':y.full}</td><td>${v>=7?(y.empty===null?'首次，无需等':y.empty):y.done}</td></tr>`;}
 $('timeline').innerHTML='<table><thead><tr><th>输出轮</th><th>kt</th><th>stage</th><th>full phase</th><th>'+(v>=7?'旧 empty phase':'MMA done phase')+'</th></tr></thead><tbody>'+rows+'</tbody></table>';
}
$('nk').value=v===1?1:v===8?7:3;$('nk').disabled=v===1;$('tile').max=cfg.total-1;
$('consumer').disabled=cfg.C===1;$('peer').disabled=cfg.G===1;$('ei').max=cfg.BN/64-1;$('ei').disabled=v<=3;$('round').disabled=v<6;
for(const id of ['nk','kt','tile','consumer','peer','ei'])$(id).oninput=draw;
$('round').oninput=()=>{const worker=(+$('tile').value)%cfg.workers;const r=bounded('round',Math.floor((cfg.total-1-worker)/cfg.workers));$('tile').value=worker+r*cfg.workers;draw();};
$('next').onclick=()=>{$('kt').value=(+$('kt').value+1)%Math.max(1,+$('nk').value);draw();};
$('query').oninput=renderCode;
