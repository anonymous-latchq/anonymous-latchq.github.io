"""LatchQ motion graphic: comparison -> sensitivity -> argmin -> quantize.

python scripts/render_animation.py [--stills | --verify]
The FP16/2-bit attention comparison motivates the sensitivity approximation.
H_d is computed from original attention and squared queries, not their difference.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import math
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'static'/'videos';IMG=ROOT/'static'/'images';WORK=ROOT/'work'
W,H,FPS,DURATION=1920,1080,30,60
STARTS=[0,8,28,50]
BG='#FAFBFC';INK='#26353B';MUTED='#64747B'
BLUE='#4586AE';GREEN='#248570';GOLD='#C7A04F';RED='#CB8777'
KEY_PALETTE=['#BEDCEB','#92BFD9','#669EBD','#397EA7']


def rgb(c):return tuple(int(c[i:i+2],16) for i in (1,3,5)) if isinstance(c,str) else c
def mix(a,b,p):
    p=max(0,min(1,float(p)))
    return tuple(round(x+(y-x)*p) for x,y in zip(rgb(a),rgb(b)))
def ease(p):
    p=max(0,min(1,float(p)))
    return p*p*(3-2*p)
def lerp(a,b,p):return a+(b-a)*p


@lru_cache(None)
def font(size,bold=False):
    name='arialbd.ttf' if bold else 'arial.ttf'
    fallback='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for path in [Path('C:/Windows/Fonts')/name,Path('/usr/share/fonts/truetype/dejavu')/fallback]:
        if path.exists():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


rng=np.random.default_rng(18)
KEYS=rng.normal(0,.75,(6,6));KEYS[:,2]=[-1.9,-.45,-.1,.25,.62,2.15]
QUERIES=rng.normal(0,1.1,(6,6))[[0,2,4]]
COLD=rng.normal(0,.6,(4,6))


def softmax_attention(keys):
    logits=QUERIES@np.vstack([COLD,keys]).T/math.sqrt(6)
    a=np.exp(logits-logits.max(1,keepdims=True))
    return a/a.sum(1,keepdims=True)


ATTN_FULL=softmax_attention(KEYS)
ATTN=ATTN_FULL[:,4:]
QENERGY=QUERIES**2/6
ENERGY=np.array([ATTN[t,:,None]*QENERGY[t,None,:] for t in range(3)])
EVIDENCE=ENERGY.sum(0)
TRACE_ORDER=[2,0,1]

# A baseline two-bit reconstruction for the attention comparison only.
# Its distortion motivates H_d; it is not used to compute H_d.
QKEYS=np.empty_like(KEYS)
for j in range(6):
    offset=float(np.float16(KEYS[:,j].min()))
    scale=float(np.float16(np.ptp(KEYS[:,j])/3))
    QKEYS[:,j]=np.clip(np.round((KEYS[:,j]-offset)/scale),0,3)*scale+offset
ATTN_QUANT_FULL=softmax_attention(QKEYS)
ATTN_QUANT=ATTN_QUANT_FULL[:,4:]
ATTN_MAX=max(float(ATTN.max()),float(ATTN_QUANT.max()))

X=KEYS[:,2];WEIGHTS=EVIDENCE[:,2]+EVIDENCE[:,2].mean()*.01
RATES=[0,.03,.06,.10,.15]
CANDIDATES=[]
for alpha in RATES:
    for beta in RATES:
        lo=X.min()+alpha*np.ptp(X);hi=X.max()-beta*np.ptp(X)
        scale=float(np.float16((hi-lo)/3));offset=float(np.float16(lo))
        codes=np.clip(np.round((X-offset)/scale),0,3)
        recon=codes*scale+offset;errors=(X-recon)**2
        CANDIDATES.append(dict(lo=lo,hi=hi,scale=scale,offset=offset,codes=codes,
                               recon=recon,errors=errors,score=float(np.sum(WEIGHTS*errors))))
BEST=min(range(25),key=lambda i:CANDIDATES[i]['score'])
SCORES=np.array([c['score'] for c in CANDIDATES])


def keycolor(i,j):return mix('#CBE9F4','#599CC0',.2+.65*(KEYS[i,j]+2.2)/4.4)
def evidencecolor(v):return mix('#EDF4ED',GREEN,(max(0,float(v))/EVIDENCE.max())**.62)


class Canvas:
    def __init__(self):self.im=Image.new('RGB',(W,H),BG);self.d=ImageDraw.Draw(self.im)
    def text(self,x,y,label,size=26,color=MUTED,alpha=1,bold=False,anchor='mm'):
        if alpha>0:self.d.text((x,y),label,font=font(size,bold),fill=mix(BG,color,alpha),anchor=anchor)
    def hd(self,x,y,size=43,alpha=1,bold=True,suffix=''):
        subsize=round(size*.63)
        hw=font(size,bold).getlength('H');dw=font(subsize,bold).getlength('d')
        sw=font(size,bold).getlength(suffix)
        left=x-(hw+dw+sw)/2;baseline=y+size*.32
        self.text(left,baseline,'H',size,GREEN,alpha,bold,'ls')
        self.text(left+hw,baseline+size*.19,'d',subsize,GREEN,alpha,bold,'ls')
        self.text(left+hw+dw,baseline,suffix,size,GREEN,alpha,bold,'ls')
    def box(self,x,y,w,h,color,r=4,alpha=1,outline=None,line=2):
        if alpha<=0 or w<0 or h<0:return
        color=None if color is None else mix(BG,color,alpha)
        outline=None if outline is None else mix(BG,outline,alpha)
        self.d.rounded_rectangle((round(x),round(y),round(x+w),round(y+h)),r,color,outline,line)
    def tile(self,x,y,size,color,alpha=1,outline=None,line=2):
        self.box(x-size/2,y-size/2,size,size,color,3,alpha,outline,line)
    def line(self,points,color='#C9D5D9',width=2,alpha=1):
        if alpha>0:self.d.line(points,fill=mix(BG,color,alpha),width=width,joint='curve')
    def dot(self,x,y,r,color,alpha=1):
        if alpha>0:self.d.ellipse((x-r,y-r,x+r,y+r),fill=mix(BG,color,alpha))
    def arrow(self,start,end,color='#A2B9B1',alpha=1,width=2):
        self.line([start,end],color,width,alpha)
        angle=math.atan2(end[1]-start[1],end[0]-start[0])
        for d in [-.55,.55]:
            self.line([end,(end[0]-11*math.cos(angle+d),end[1]-11*math.sin(angle+d))],color,width,alpha)
    def curve(self,start,end,alpha=1,color='#AFCCC1',width=2):
        p0=np.array(start);p3=np.array(end)
        p1=p0+[(p3[0]-p0[0])*.5,0];p2=p3-[(p3[0]-p0[0])*.5,0]
        points=[]
        for u in np.linspace(0,1,32):
            p=(1-u)**3*p0+3*(1-u)**2*u*p1+3*(1-u)*u*u*p2+u**3*p3
            points.append(tuple(p))
        self.line(points,color,width,alpha)
        return points


def query_tiles(c,q,cx=487,cy=528,size=57,alpha=1):
    for j in range(6):
        x=cx+(j%2-.5)*74;y=cy+(j//2-1)*76
        color=mix('#D8EDDA',GREEN,.18+.65*min(1,abs(QUERIES[q,j])/2))
        c.tile(x,y,size,color,alpha)


def hot(c,q,phase=0,alpha=1,show_query=True):
    c.box(908,282,430,516,'#E3F3F9',18,alpha)
    for row in range(6):
        yy=345+row*78;strength=float(ATTN[q,row]/ATTN.max())
        path=c.curve((640,540),(884,yy),alpha*(.25+.5*strength))
        if phase>0:
            px,py=path[min(31,int(phase*31))];c.dot(px,py,3+strength*2,GREEN,alpha)
        for col in range(6):c.tile(964+col*63,yy,52,keycolor(row,col),alpha)
    if show_query:
        query_tiles(c,q,alpha=alpha)
        c.text(487,703,'Observed query',27,alpha=alpha)
    c.line([(571,452),(599,452),(599,604),(571,604)],'#B3CBC0',2,alpha)
    c.dot(640,540,4,GREEN,alpha)
    c.text(1123,850,'FP16 keys',28,alpha=alpha)


def observe(c,t):
    p=min(2.999,max(0,t-.8)/2.18)
    hot(c,int(p),p%1,ease(t/.8))


def attention_bars(c,values,cx,base,width=490,height=155,color=GREEN,alpha=1,reference=None):
    step=width/6;barwidth=step*.55
    c.line([(cx-width/2-12,base),(cx+width/2+12,base)],'#CFDADB',2,alpha)
    for i,v in enumerate(values):
        x=cx-width/2+i*step+(step-barwidth)/2
        h=float(v/ATTN_MAX)*height
        c.box(x,base-h,barwidth,h,color,2,alpha)
        if reference is not None:
            old=float(reference[i]/ATTN_MAX)*height
            c.box(x-3,base-old,barwidth+6,old,None,2,alpha*.75,'#97B2A6',2)


def paired_attention(c,q,cx=600,base=654,width=402,height=183,alpha=1):
    step=width/6
    c.line([(cx-width/2-8,base),(cx+width/2+8,base)],'#CFDADB',2,alpha)
    for i in range(6):
        x=cx-width/2+i*step
        a=float(ATTN[q,i]/ATTN_MAX)*height;b=float(ATTN_QUANT[q,i]/ATTN_MAX)*height
        c.box(x,base-a,step*.35,a,GREEN,2,alpha)
        c.box(x+step*.41,base-b,step*.35,b,RED,2,alpha)
    c.tile(cx-116,base+70,16,GREEN,alpha)
    c.text(cx-97,base+70,'FP16',23,GREEN,alpha,anchor='lm')
    c.tile(cx+38,base+70,16,RED,alpha)
    c.text(cx+57,base+70,'2-bit',23,RED,alpha,anchor='lm')


def draw_hd(c,values,cx=1280,cy=550,pitch=60,size=47,alpha=1,highlight=False):
    extent=5*pitch+size+30
    c.box(cx-extent/2,cy-extent/2,extent,extent,'#F0F6F1',18,alpha)
    for i in range(6):
        for j in range(6):
            c.tile(cx+(j-2.5)*pitch,cy+(i-2.5)*pitch,size,evidencecolor(values[i,j]),alpha)
    if highlight:c.box(cx-.5*pitch-size/2-6,cy-2.5*pitch-size/2-8,size+12,5*pitch+size+16,None,6,alpha,GREEN,2)


def trace_state(t):
    starts=[12,14.45,16.9]
    total=np.zeros((6,6));q=2
    for k,start in enumerate(starts):
        fraction=ease((t-start)/2.1)
        total+=ENERGY[TRACE_ORDER[k]]*fraction
        if t>=start:q=TRACE_ORDER[k]
    return q,total


def trace(c,t):
    # First focus: the same query, with FP16 versus reconstructed two-bit keys.
    # Only the six hot-key attention entries are shown; older keys stay in the
    # softmax normalization for both comparisons.
    if t<9.5:
        transition=ease(t/1.8)
        if transition<1:hot(c,2,alpha=1-transition,show_query=False)
        leave=1-ease((t-8.6)/.9)
        query_tiles(c,2,cx=lerp(487,430,transition),alpha=leave)
        c.text(lerp(487,430,transition),714,'Same query',28,alpha=transition*leave)
        a=ease((t-1.5)/1.2)*leave
        c.curve((555,540),(749,435),a,'#ADC3B8',2)
        c.curve((555,540),(749,764),a,'#ADC3B8',2)
        c.text(1050,271,'FP16 keys',32,GREEN,a,True)
        c.text(1050,603,'2-bit keys',32,RED,a,True)
        attention_bars(c,ATTN[2],1050,480,color=GREEN,alpha=a)
        appear=ease((t-3.5)/2)
        attention_bars(c,lerp(ATTN[2],ATTN_QUANT[2],appear),1050,811,
                       color=RED,alpha=a,reference=ATTN[2] if t>4 else None)
        c.text(1050,901,'Quantization changes attention.',30,INK,a)
        return
    # Second focus: use the attention comparison to motivate sensitivity.
    # The concise formula states how H_d is actually computed.
    a=ease((t-9.5)/.85)
    q,total=trace_state(t)
    paired_attention(c,q,alpha=a)
    c.text(600,348,'Attention sensitivity',31,INK,a,True)
    c.arrow((858,555),(1000,555),GREEN,a)
    c.text(928,511,'Estimate',24,alpha=a)
    draw_hd(c,total,alpha=a,highlight=t>19)
    c.hd(1280,294,43,a)
    c.hd(1280,823,31,a,False,' = Σ a · q²/d')
    c.text(1280,880,'Accumulated over observed queries',25,alpha=a)


def candidate_xy(i,cx=1270,cy=530):
    return cx+(i%5-2)*66,cy+(i//5-2)*66


def selected_candidate(t):
    if t<4.2:return 0
    if t<6.3:return 24
    return BEST


def pipeline(c,t,alpha=1,include_codes=True):
    # A single selected candidate is carried into the quantizer.
    emerge=ease((t-12)/1.5)*alpha
    key_alpha=ease((t-12.8)/1.5)*alpha
    choose=ease((t-10)/2)
    start=candidate_xy(BEST,960,530)
    tilex=lerp(start[0],960,choose);tiley=lerp(start[1],331,choose)
    c.tile(tilex,tiley,48,GREEN,alpha)
    c.text(960,264,'argmin J',38,GREEN,emerge,True)
    c.arrow((960,373),(960,469),GREEN,emerge)
    c.arrow((600,550),(783,550),BLUE,key_alpha)
    c.arrow((1137,550),(1295,550),BLUE,key_alpha)
    c.text(486,735,'Keys',30,BLUE,key_alpha,True)
    c.text(1402,735,'2-bit keys',30,BLUE,key_alpha,True)
    progress=ease((t-15)/4.5)
    for i in range(6):
        sx=450+(i%2)*72;sy=458+(i//2)*72
        dx=1370+(i%2)*65;dy=466+(i//2)*65
        c.tile(sx,sy,48,keycolor(i,2),key_alpha*(1-.8*progress))
        if include_codes and t>=15:
            xx=lerp(sx,dx,progress);yy=lerp(sy,dy,progress)
            convert=ease((progress-.48)/.16)
            color=mix(keycolor(i,2),KEY_PALETTE[int(CANDIDATES[BEST]['codes'][i])],convert)
            c.tile(xx,yy,lerp(48,43,progress),color,alpha)
            c.text(xx,yy,format(int(CANDIDATES[BEST]['codes'][i]),'02b'),19,'#FFFFFF',convert*alpha,True)
    # Draw the operation above the travelling tiles, so they pass through it.
    c.box(810,480,300,140,'#FFF1BE',16,emerge)
    c.text(960,550,'Quantize',43,INK,emerge,True)


def clip(c,t):
    # Keep the completed H_d matrix at the boundary, then move it to the side.
    handoff=ease(t/2)
    if t<.8:
        old=1-ease(t/.8)
        paired_attention(c,TRACE_ORDER[-1],alpha=old)
        c.text(600,348,'Attention sensitivity',31,INK,old,True)
        c.arrow((858,555),(1000,555),GREEN,old)
    if t<9.6:
        vanish=1-ease((t-7.2)/2.4)
        cx=lerp(1280,440,handoff);cy=lerp(550,530,handoff)
        pitch=lerp(60,42,handoff);size=lerp(47,32,handoff)
        draw_hd(c,EVIDENCE,cx,cy,pitch,size,vanish,True)
        c.hd(cx,cy-5*pitch/2-88,39,vanish)
        arrow_alpha=ease((t-1.5)/1.1)*(1-ease((t-7.2)/.65))
        c.arrow((650,530),(1005,530),GREEN,arrow_alpha)
        c.text(820,478,'Weighted error',29,INK,arrow_alpha)
    grid_alpha=ease((t-1.5)/1.2)
    grid_center=lerp(1270,960,ease((t-7.2)/2.4))
    chosen=selected_candidate(t)
    argmin=ease((t-6.6)/1.2)
    others=1-ease((t-9.8)/1.7)
    if t<12:
        c.text(grid_center,298,'25 candidates',32,INK,grid_alpha*others,True)
        for i in range(25):
            if i==BEST and t>=10:continue
            x,y=candidate_xy(i,grid_center)
            score=(SCORES[i]-SCORES.min())/(SCORES.max()-SCORES.min())
            checked=(t>6.3 or i==chosen)
            color=mix('#E2EADB','#98B79C',1-score) if checked else '#E7E8DA'
            if i==BEST:color=mix(color,GREEN,argmin)
            opacity=grid_alpha if i==BEST else grid_alpha*others
            c.tile(x,y,47,color,opacity)
            if i==chosen and t<7.5:c.tile(x,y,57,None,grid_alpha,outline=GOLD,line=3)
            if i==BEST and argmin>0:c.tile(x,y,59,None,argmin,outline=GREEN,line=3)
        if t<7.2:
            c.text(grid_center,779,f'J = {CANDIDATES[chosen]["score"]:.5f}',31,INK,grid_alpha,True)
            c.text(grid_center,830,'Lower is better',24,alpha=grid_alpha)
        elif t<10.8:
            label_alpha=1-ease((t-9.8)/1)
            c.text(grid_center,792,'argmin J',45,GREEN,label_alpha,True)
    if t>=10:pipeline(c,t)


def store(c,t):
    clear=1-ease(t/1.8)
    if clear>0:pipeline(c,22,clear,include_codes=False)
    progress=ease((t-.6)/5)
    c.box(503,406,914,316,'#FFF2BE',22,progress)
    for g in range(3):
        for i in range(6):
            c.tile(573+g*181+(i%3)*39,478+(i//3)*56,29,mix('#E5DBB2',KEY_PALETTE[(i+g)%4],.4),progress*.75)
        for j in range(2):c.box(564+g*181+j*47,591,33,12,'#C8BE98',3,progress*.75)
    for i in range(6):
        xx=lerp(1370+(i%2)*65,1116+(i%3)*39,progress)
        yy=lerp(466+(i//2)*65,478+(i//3)*56,progress)
        size=lerp(43,29,progress)
        c.tile(xx,yy,size,KEY_PALETTE[int(CANDIDATES[BEST]['codes'][i])])
        c.text(xx,yy,format(int(CANDIDATES[BEST]['codes'][i]),'02b'),max(12,round(size*.4)),'#FFFFFF',bold=True)
    for j in range(2):c.box(1107+j*47,591,33,12,BLUE,3,progress)
    c.text(1163,654,'FP16 scale + offset',22,BLUE,progress)
    c.text(958,811,'2-bit key codes',32,BLUE,progress,True)
    c.text(958,875,'Same bit allocation · same storage format · same cache schedule',28,INK,progress)


def frame(t):
    stage=max(i for i,start in enumerate(STARTS) if t>=start)
    local=t-STARTS[stage];c=Canvas()
    [observe,trace,clip,store][stage](c,local)
    c.text(960,108,['Hot-cache observation','QTrace','QClip','Cold-cache storage'][stage],76,INK,ease(t/.65),True)
    # Deliberately no corner caption or watermark.
    if t>DURATION-.7:c.im=Image.blend(Image.new('RGB',(W,H),BG),c.im,ease((DURATION-t)/.7))
    return c.im


STILLS=[5,11,14.8,18.7,22.5,27.5,31,33.5,36.8,40.5,46.5,58.2]


def verify_math():
    assert len(CANDIDATES)==25
    assert np.allclose(trace_state(20)[1],EVIDENCE)
    assert np.allclose(EVIDENCE,np.einsum('ti,tc->ic',ATTN,QENERGY))
    assert np.allclose(ATTN_FULL.sum(1),1) and np.allclose(ATTN_QUANT_FULL.sum(1),1)
    assert not np.allclose(ATTN,ATTN_QUANT)
    assert BEST==int(np.argmin(SCORES))
    for c in CANDIDATES:
        assert np.isclose(c['score'],np.sum(WEIGHTS*(X-c['recon'])**2))
        assert np.isin(c['codes'],[0,1,2,3]).all()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stills',action='store_true');ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    for path in [OUT,IMG,WORK]:path.mkdir(parents=True,exist_ok=True)
    verify_math()
    if args.verify:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video=str(OUT/'latchq-overview.mp4')
        info=subprocess.run([ffmpeg,'-hide_banner','-i',video],capture_output=True,text=True).stderr
        assert '1920x1080' in info and '30 fps' in info and '00:01:00.00' in info,info
        assert 'Audio:' not in info
        subprocess.run([ffmpeg,'-v','error','-i',video,'-f','null','-'],check=True)
        for i,t in enumerate([14.8,23.5,36.8,46.5]):
            subprocess.run([ffmpeg,'-y','-v','error','-ss',str(t),'-i',video,'-frames:v','1',str(WORK/f'encoded-stage-{i+1}.png')],check=True)
        print('Verified: 60 s, 1080p, 30 fps, silent H.264; attention comparison, H_d, argmin and all frames valid.')
        return
    sheet=Image.new('RGB',(1920,1440),BG)
    for i,t in enumerate(STILLS):
        im=frame(t);im.save(WORK/f'focused-{i+1:02d}.png')
        sheet.paste(im.resize((640,360),Image.Resampling.LANCZOS),((i%3)*640,(i//3)*360))
    sheet.save(WORK/'focused-storyboard.png')
    frame(14.8).save(IMG/'latchq-overview-poster.png',optimize=True)
    frame(14.8).resize((1200,675),Image.Resampling.LANCZOS).save(IMG/'social-preview.png')
    print('Focused storyboard ready; attention, H_d and candidate calculations verified.',flush=True)
    if args.stills:return
    import imageio_ffmpeg
    target=OUT/'latchq-overview.mp4';temp=OUT/'latchq-overview-rendering.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
         '-s',f'{W}x{H}','-pix_fmt','rgb24','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium',
         '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-map_metadata','-1',str(temp)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for n in range(DURATION*FPS):
            proc.stdin.write(frame(n/FPS).tobytes())
            if n%(FPS*5)==0:print(f'Rendering {n//FPS}/{DURATION}s',flush=True)
        proc.stdin.close();result=proc.wait()
        if result:raise RuntimeError(f'ffmpeg exited {result}')
        temp.replace(target)
    except BaseException:
        proc.kill();raise
    print('Exported simplified 60-second overview.',flush=True)


if __name__=='__main__':main()
