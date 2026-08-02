#!/usr/bin/env python3
"""Capture the immutable V2.4 A_ONLY artifact root."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Callable
import zlib


HERE = Path(__file__).resolve().parent
COMMON_SHA256 = (
    "8a68a6ef76806c311e7c5d7fdc2b7fe8"
    "13230b846254097bd1d3a4a47c254bc2"
)
COMMON_SOURCE_B85 = b"c-qZ8{dd~NvcL0JZ1n15XaR%m#3?C#k6;{guMIwc)8-PMj*u2xwUDSJoVYgs`<vMhX|)pAZF=u{xwl5zot>T8nVqj)_^($7>nJ>!yZ!<7@5$;Rz771Nl3^I*TPvgvSy--Dv%MgK=T#WYX+#!5Na9;cJ|5POi514~!m?vxTdQ~-QbPSWd{_mpA4hdSQz|XOU`b4KvBqxGB<^w*gfX#vKZvc^4g9E75^atZ)7V{7`F?9fx1KwfukOlnoRCJ5Z1^n-eEA$ie5REZ;}|*36}(E2FhLAx>DS~1$Vi^obKufe=1}R0A68UM1COqMBWLOO4^`5&JkOeYv`VhRAP(%nE0s#;y-|;}85w0*1lNNlsMLXpz`LjAN__>Z?8l{#hsOYgz_Mk0^dq!DyKLxOPtJ#<{^XN+)tQ|0K5{?|d(|+*rb+Kzhx#@(?{Ub3p}>#NJBP2|z=(2*z(1pL)Lhsn_Uof#dSW+E=;FktN7Oz%X*3T{ju&qX4o2U;rPkuenjhNDxAPw@%Yugao1^2y6Z_=IIx$L>Quq3_V~)q?=A<+F(3`*<#vixttrh%F^W?C8*r=Zz*WbQ2N>}H@LC@@*zBjLiBLsWhIC=}MpT_-e=fWKKM*Ysk7}_8C+`+1OXtap&+M*}c@sVRxllIrxe$;Kg?p!w-huw<?{)7HasXH8;^+%UxXEf=bb-Ik2(e=QDg-Y7Y(Qr61JLYh3@ySR=oQ-<pa~1roGoJ8PbJXjc_6NQ3I2m=>`PJ+X5Pi8xe)@^LIjWEvX_h*J)6uYhYEJr>J(Gz&q;)%3u0ZO1=|iLZ-&a4)t%&~o*?BzP)Zp)-_)FN|mh!h^h(Vw0IBXHZK7;*&Lz~qfHx+`de*`}LdHO?b#t>FeD5a9^Sy2QE$+l!K41=)z(el>(vC?8_R{)ZLj@G_U!?ILd0pidF3D-WS9sfntTU3)bE4;aH5zmd{X%vUEwB!5q5w}mUj$35m1y;=4VYPl%c3Qm)OXSg*I#;}5I)8|1G;1YD>h*d;={On3Y5xC3X-HwkeRW8NDoIN<(;`f#3A3-(aeET3sRnRM?Ys7w<waVDpt&tR(@s<ZwkGuVwHwm19r%um%F`nAAn>YW3A9-^v<22LWLyqFk2oLhf`Dlz>5!B=%Z+GOtppx9AN>kCTE0hJYjLYczpkhaD<)nSF7=$EStX^ZJiyp;Ocmlr{8b{WfctO=>m==VGDph)!swYFI~Uiz7I|b-ZdwTxd6ZAD!cCRj0I!b>6}$>NBiR8jMsHwWs0^hXz$&`FyIe0@2ngG~!FYn_m$8I{kPm@{{L~gK!Go|dS&pEg_n|jRcuJ^MRF7duq;EN96s$v=meDV^m>)`IVn(+Xcm~0YCK{&NoRNiR-9*4&5Y>m~=oGCWdE#%~i(&8AURNp(cx)?<!*UQ+i7_+>!?WSV#qbxSk}45L_9U=L2-XdeLkj26$PV3A4ABOI4OY}=TvRy<&NcmN(^X7{W2TP8iU`GrojQw4IUZh*x;=Av)f*@(^2i2nj0#~+p1tKMn^OpZD5q+H(L*_<wW62;E`0>3kInvg)ca88L+ep&E(3>Fszjj$`mZlKigXoR))$x#C1Qo4^-xZGzTARyA#*RV??{VLghE#cs*+{{0)_9{hav`nwobps)OX4ZLyFc~#5CN+;llMH+))+<AX%80gT$~TcH;-Y=2^?RV-c~N(tW`ia8Vqpn8u{l^@D;|T)=|n$o-?Jd5I>{^rM0<JJTiA3hkIAz08&aq?I&qIcDWI8FnYbYh8RClLxvBs|^EE4Pcb22*&+?^@@j!K)BBLE5lU3rC%NQhDLF@lH1XUR^Sn1jEs?Bdz+yVmI7R!V_Iv7{zGJs82}dT)-MBR?a>K53!}zVQMD}Dz_d#Az!NM%f?1HUWJm#zK1bPzj()@t)|5`McxYY5(traDmPUhII%=0?_yuHFiy(nGgfrkF!R)wp3{kqeV7yN(mW_J_6sD}3dj^WyROcW!G#0v5X02c>yZ<yVhlAl{IOuoHQRkP$QLlZI<yRt5<gty2%eZD11~ElZi{hb5x+yRANr;-cB*dN?0q{*2E%5G4Gwl7c=FMQb)GOEB$ck*&g(bptk-Ih5XFOE0P7M-=kHsD>V{{>9T6^eZ`O>hsiiJ{o5R~x?N+34j-Mz0X7ZckTdkT!^hu0HjW9rU<?@>-*ooj)O#)&le_U`@KpA_$FF(!%~zy`L#FoZ#voc2aJPatK-7Cs${Jfi@ZlvAN)0TR4fR<^CP7*=;Ea#A*!FpbR8Ui&Wgoid0&>3uunZoj`{Ma5Ri&;lD})Vmx`dggft(z2^hu#X3rvl+%`tzb~a!ax;b5OiJD=pibt+VHoI`Z8_KN(-l*1Qb;zsCiODnkU%3GL}(8nQEg_De)zX8tueEE*w}L5jtmFe$+_ZeM_vz_#ukvviH@E%SR0ucH7nrJhXAgj=2D?=DaOE`@*R<8^>>7pS(e7hzaKq)<f?dwA|L$E3BiG=g=QDH3IM2!UspKawI)&B%N%O_|+(JP*KKvMykW)4Br8Vl~--}XcjWq1h7>IR<evzsXOcCDk?u3!Y#FKGmgwaQjON5A?Gk!Q8VMD4WQVysp&eh3t2FLe22;w<i%Jj4U8vaso_kxGLTc*iqD%(H3EGy@90Cs^RbAotk8m>CTf?BD*UfyR6q~XFer(CaGhD<OK9tO(wP)%>wij1U!tVX>S&P&^Rn75s+^>BK^4lWP)z?0DfI^*dxObv^y!773WFfdNK33LOn^BG`oB3uSO8TaU>Vb9&-L%p_UsVH4S5$(nv=qD0~zsX)Sk{tbIZXGAjiIIPjSqw#4QGEg7Nk`#Ij+Vz1LU4iWwC*zQy?>8{7m;CV(o;a{F=U@-P8iHml>J%U0T1akb5Q7|Gb=`inp`aTN<ZFZg0c4@=}U;#CXgIADbmojdK1$~ZDrJ7)JevDRvZDGAQN22xKQit1W0lr2rc&`89suwvQX0)++`P%n@-g@nj*z#K8GNgI2syj1I$da>Xxif&C_S~pQ$$}#+<u}Hgl6fItOmM&B9B-Z0BKrFjN_YSHl3iYIBlgADG->8GkmazP+vR(gu9e@Ymn8Ed{jg0wBY&%W7wE=`0X>?}P0`={^72CHEs?+uZtjSieZUFI_9gA?Q{aWece5db%A>J?(i)A@;zDl_9SY-tmprLDda=}=7xpkE!R|X)0aLTA%89eewd>HT7B12+?x9FC7o*}U&MEFh^xRBz2)|rCLav~f^1by3EJJjTkix<^+u`m|V4_j@NRc`U6rpG3iSyGs1%XixQ*xqxV`h#J6&wav_u)P;OT@Fv-bMa)PfS0VyJZ|rOdSZ|5y@#i+Z2K2~Z14SHs9EjO#n~3jax>k@s7p#xWkYH;^2mAEa1<<t$10GjFBH(wwV_7z5QIYGA|0LUd_`&(McoG5k_)o`zIQep^*(?0_my5GUTTV{Hfd7GruLC6ZDdLw)65$uN=|#)|Iiyx+WYSa4^E=>G9i;Zfkg$roPDwn+%=7A4pD-qb2jOXH0twQ$bU1XXshV2@s1ev-vZa?DOMyCMH60K4Dgt4W-60Os@%%u)w${(=D1U)$}m*yoMb#6S}egXGK~l!zz-chkwBA8M>3X~flawjpNXsk(%o&vt%9PZ3C1*{CK_hU+FLxY`F8F^wK&gZcD_OuwNw|;bJqSb9uCam`+xMhlb5q5jSI##Qh?mDQ7pPg!sux{C3S`++F@WsJ8r5k)}F`u1f>}c_#mJiBMz*(bNW>rIs<r21=Gkw*?RH-6s;KT%Eo?V6Vz|QV7)5u<IsIDBH30vs)AN^$M1cr?JaA24%s_z?OnF^#sn;uS_6=WUO<S7%aH9xrSAQ<hTP#tu;q<p=|!yoJv`D<M{+?=mk<S2uoh}I8V&g$`GLVnjQRYW92{gvB$CS1*?3b{`W!*A_}T7abyvQS5HATu$nM4heJ?Kaq*@RB2(8B`1sx3f74(MSSMUS<3VxzrVV~S_B#-#eP21R`;0&0E5hapak5(RprpA+2q3~F#9^fi?JcDmYM<uIsEHOIk_byJyy6Ppq=5fPWg@DwBK)8zhU{LypyW$hDmPn~<uR^+TzcMfP2wj`Dk*mlfCbVrZ90TGqAF`FYgt$6GsFdY7nNw(;p|SzmDdU{uYt4!dHAF@vlv7|}(wU`*S&pQEpyU(~On@{9CR7DvnMcCpJ*$%D8wHHzS3-Stc%cIdR(l6{Hw!K*Md{J=pQsS2V#u9)6%>rbj(<#TOPZDz8s3Z(&M;uiwBuKm<7`*D{TYyA+LS7%ng^#Ep!AEBn2TL$=b15CYOQ~PugfC7mM`IJxq~mBkCXJsWuw|YEmGkkQv7)0X$!5RyjWD0rw%d@`18h*@2Gsho-><iuz<!5l^ajUv>GV4c?_VeSv@ugGc@)zZOqX689bUb3e3xSuXEbV#YCc#ur0V%Z7iFI+1SGZ^cQ$-rSLeB3+OQ;a#7BC&P%EPIGfE`KFm#pTo7+d)#68CT9BMlxhZXD_w`HJP0nh!vz2g>QB9r70#Q#7XQ~@9ME+oCo`g5+gt0pvm|?VuM8=#)ibeR3T)ZlZr4qr`vp#Y1$dDfacLvK#3>BW8Ynr#za%d<<$&=sb*6VoNdNBkU5?G5o`4np-uVZ!3JA)5BSfuMwfAE2o_O}GmX~~$RtV^Y3Q*7k<T;O@sj&&1Kbfos28G-zf-DOlPNm>Qqi99PLK+K++k@R%#`(4JT==kpCxtTSO0&fk*73q=pA`HoL{ZUfBVU@bnw8(a5n(Vb>8L4UE1tSoC!9hzd-Db5lk}q;=mlm{x&7@P2LNdKpCetn#%2FAxl2r<CLg`+tXgwZrxSEA%0bQ#6X-=5Y^c_f3RLO0lcB;8jUpL!UQ`FQqrJNm=VrM&w)gEO=o?L!<DP=_!gejkR_F_(1Z9qY~qZ+fzh~>P4q=qlptjK-rB1)|D7LSri$yIZnYzjy%#D*3}w}aEGA%vfJHAqgDv26*oewP(wy}EsfT-)+Y0VM6&wtb=RiKG&mVJCZ)R@*^*qN$`+TfePUO}(I5DUA1Wd+Y`KD`UcUhInFWs-+z<#0xicmYM98zGQaSzo&j2gb&F|;4w;CZ`obRWydrsZ{=K7^c^vBWl<+7D83{4T4;8}!u3;dQd`0+WK*}KNouUpNS+8DM8Jnq73POP{j)K@qeE&TvzG6O6R2RYT{%G)<g@V|LVPyf;jMn3Bt_WHX?riiGNuZm;9$oxk}VxeYa8fd>x5_hi{5iHf{olw-@-*!eQIQcE+c|O=6`QUkSNXzLV`k%uN`4L<Lq%dG*1aP>Sn;smwEfY-#>3TORcg6dYQv6<mnN|e~ZW7g|+e`XPeKevy}70LooXM#BlSid?^>Hrwlc?=IM+(rP1|()ma9^L0VQ%u(oo1F_K7I`c_UMPcG8)(QkE!-{cbik~$W<$LCz+3*00Nl6JcZO_~2ax-I24O^=8`6S56h8O4MQMfF^<A{&x{b>JmjC4*%q*w6rTr|US3=c*E3DGw5PV4p*d2D=Eiax$M{7Qi{gs=7W?LX_JiHAh$>*r~ce%3s@N#n&qNeSyMzm4EdDB}bV-CDf!%#4QB~Kg7)1P0QSZ6!B9KwlQSjNx8j;Oz(Ouug8c`rzD;7`H&x=-Ud<ZTT7aByvMIy^-r<SwqkOCw=xf+hv?vrhCcNULf|A=9zbRcUfL;HU+&2M!T*lGSgT18TVCtnF~-mKcZVC^d=^p)tqb_TMzBneJSsBAzjUt5>v8YY{O9|&LD;w&!fp1&yi2;o8zd$F-gR877A;*;tNp&F;Y00awQjH1u2WsvZhgLvuvA~G@l)}(9o<*`fR{_)VePwd1mPDFTRYpYl6|s27`59?Ny&$+YxDXVxHi6Pz9XXXs@pf;Uyn`J%n5fF<-zDxyIG-M-IyFVH~akZ2)ARqSDf2cT-O^K19vNMwsmgc{NS%HzZ`Zo8lGNvCmMG6bk-1pX9SEV3m01ev4HI*kf*(m{cdjyC@+sEuvqyYopk{Gi)Z-_2=+?uiqwutEg<-1P>9KcFIZD2te-@7u%hI~3+9$5BqKz^PpZB<u3A;&z^J?v(zqC+gX&*>JZ>A-v^gu1#}UXwDC8wwcJ3{gea{oLPawJMU7DSbo&H7V{Y8&y9SBxew*QPuv#U%G5k}UavS2PlT+c*FTR_5ucabq<f$In+youKH^1*c4idL3QTeI0g)iA2x9aa*Zot*bZm*~iuDk6IjX@mYdZP066B)M#_SgTmKL#f9i@rok%#$SR9jQ)nAzE&wy@BiRAgC2Zm&l3N<$AlmdPkqHjuWAx^GB1ZGk?gB-Wut8rd;>@hV^KKBFS(-)Sov4k@K}=<{fRs$L5rO|f7}(S#`J5P+|e+$9awVQ^1xE9gKRym_VGH`jTO81*}DqcZYN}pI~j`1S?4z&48G(nkTD~_yT@tJ4x3~`$jwdk4e6jA_%YrbXlZ5J9ob&IUU_t?fh)tanmd!X{~&uaUtBwyYG*r{i+%Z9dpj@qZ2{kL(GUOW*<nE^vnFZBY)7eyC9kaLxQ;S0N^(^oV_)QAW%9rb4z-A<d}~;qWh<f6qanN}rgze%<1+?3_?dcicC|vW)oLE7&R)lyGQ$QVSg9FHwVH~<V@TMhJb|B;wEQrda!NplGFYYtMq${@r)F=Jg3$?2c?=i|r5M;-rMN_aeF;XnV97`kSct)bC4I|i$M>vSp&wF0d}cB-TBLydEG%O<g%gXEhcP8}DQwC>7965NlShRLup)mQK4SN`%?P}^xdt<2GIwXFUw%w7&Mo#9FLtOZm8>ZuH7eqpo!ipUjdCvU)A&me-euNZvo%>asn!9?{5Kh@;x`x+*Z7RqCQ0_+km+lQJxLXnN4&T=YS8?$yj;n~GX^G`jjwJ4pXzodUuw!D)00fCVjq_oW{uRx{Pohr3gx?aNL|#E1kgpDY4BUrt4FnkFc|e0b%@NNsEcPTg*~dC7xZVtPz_@9tXNg6t5sS++2!G)xM7wRQ#qlUNLAAWR!)Q<@B8E-#A@-2Jp~w&MZ>$qEFV%~w9*MzCZ;=6PWw#R5E$ZV>Zk{U#U`91R4sT0N!F0-bSUbEW{TxxhK2#-vT$M9RKbx68u^ZAKlFo<qx{jIdjIPFxWk|2bczDcREk)1nT`~rc7~ffLQZZ`aF%TGNDn*<3qj2ALS&ye_>wv|GS_|ta&sX>b#h=xZ3^iH!D?>Vch7(=A&z%#ODQ@Plqq-cR<ubhnfRvVcGX#BPz(9wOvr4<CQb0Gv+M2+o!-mm*Bs{#1A)GD$Fw4ye)-IrsoFKSz+JfJh3b?F;%C+60yR8LWF(m4S4GSOTErf+ZAt*b(rSV`P9_A=;yca-+*GtXuS&H&6X!9*uG82`dDH@`G*rtOsw|+e!7U|55KTHHH^wR;Z^^}e!X&GZvMm?)aUanticnq{A!3ULE7sN&fkjKIr5^67!QQTAPfc;MoT>rO0($|T+qkRyV}qsrWnpe|!Ddcp6H<5x2^y;4nvqxSEW~RaLdl_2gJq6-u(gW{)<WZ|30{%)JL`}s*%$D7u!ZTD%&+34<)|$`O(edhY0A@B{NW9tXm~xjx}KPw_v7Kk^`xhr2%zm}1L;>cf`~CQruS=qJQ*h!>j8@Y*js&hhhNX+FXCH<gds44JN7CQse<3gQK~P%mihxNL-UtW=Er+Rz0O3w^nUGLXus-{4sHZ*j*qwgHbbyjU|G6N8rtvDh>o9#{QZ-xzZKm-Fxr_=GCy1c1Ew5AJqKc6W`6MPZ}>?XzOLHO3W=q!7X8Z6X@At~;)zKS^O7Du{yADC)c*myS~i#"


def _load_common() -> types.ModuleType:
    raw = zlib.decompress(base64.b85decode(COMMON_SOURCE_B85))
    if hashlib.sha256(raw).hexdigest() != COMMON_SHA256:
        raise RuntimeError("E_COMMON_SHA256")
    module = types.ModuleType("s39_v24_capture_common")
    module.__file__ = __file__
    exec(compile(raw, __file__, "exec"), module.__dict__)
    return module


capture = _load_common()


SCHEMA = "s39-cp0-r1-artifact-root-v2.4"
CONTRACT_SHA256 = (
    "264d16b33d56176ee6d3ac84471b3616d4b17e5bea785e08d03723ceb73a439f"
)
CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
REQUIRED_BUNDLES = {
    "cuda_monolithic": ("cuda", "cuda_monolithic"),
    "cuda_route": ("cuda", "cuda_route"),
    "op12_stagenet": ("op12", "stagenet_worker"),
    "op15_direct_relay": ("op15", "direct_relay"),
    "op15_stagenet": ("op15", "stagenet_worker"),
}
CAPTURE_KINDS = {
    "artifact_root",
    "cuda_monolithic",
    "fast_fresh_readiness",
    "joint_phone_cuda",
}
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        capture.require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def _reject_number(value: str):
    raise capture.CaptureError(f"E_JSON_NUMBER: {value}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise capture.CaptureError("E_CANONICAL") from error


def _read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    capture.require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise capture.CaptureError(f"E_READ: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        capture.require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_INPUT_BYTES,
            f"E_REGULAR: {field}",
        )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            capture.require(len(raw) <= MAX_INPUT_BYTES, f"E_SIZE: {field}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    capture.exact(identity(after), identity(before), f"E_TOCTOU: {field}")
    try:
        value = json.loads(
            bytes(raw).decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_number,
            parse_float=_reject_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise capture.CaptureError(f"E_JSON: {field}") from error
    capture.require(type(value) is dict, f"E_TYPE: {field}")
    capture.exact(_canonical_bytes(value), bytes(raw), f"E_CANONICAL: {field}")
    return value, bytes(raw)


class _Common:
    canonical_bytes = staticmethod(_canonical_bytes)

    @staticmethod
    def sha256_bytes(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def parse_json(raw: bytes, field: str) -> Any:
        try:
            return json.loads(
                raw.decode("ascii"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_number,
                parse_float=_reject_number,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise capture.CaptureError(f"E_JSON: {field}") from error

    @staticmethod
    def read_canonical(path: Path):
        return _read_canonical(path, str(path))

    @staticmethod
    def exact_keys(value: Any, keys: set[str], field: str):
        capture.require(type(value) is dict and set(value) == keys, f"E_KEYS: {field}")
        return value

    @staticmethod
    def integer(value: Any, field: str, minimum: int = 0):
        return capture.integer(value, field, minimum)

    @staticmethod
    def text(value: Any, field: str):
        capture.require(
            type(value) is str and bool(value) and value.isascii(),
            f"E_TEXT: {field}",
        )
        return value

    @staticmethod
    def digest(value: Any, field: str):
        value = _Common.text(value, field)
        capture.require(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"E_DIGEST: {field}",
        )
        return value

    @staticmethod
    def absolute_path(value: Any, field: str):
        value = _Common.text(value, field)
        path = Path(value)
        capture.require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
        return value

    @staticmethod
    def stat_record(value: Any, field: str):
        _Common.exact_keys(
            value,
            {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
            field,
        )
        for key in value:
            capture.integer(value[key], f"{field}.{key}")
        capture.require(value["inode"] > 0 and value["size"] > 0, f"E_STAT: {field}")
        return value


AUTHORITY_SHIM = types.SimpleNamespace(common=_Common, MODEL_ID=MODEL_ID, PHASE=PHASE)


def _validate_cuda_launch(
    value: Any,
    contract: dict[str, Any],
    roots: dict[str, str],
    bundles: dict[str, dict[str, Any]],
    components: dict[str, dict[str, Any]],
    token: dict[str, Any],
) -> dict[str, Any]:
    launch = _Common.exact_keys(
        value,
        {
            "allowed_system_roots", "bundle_id", "bundle_root", "bundle_sha256",
            "command", "cwd", "endpoint", "env", "expected_capabilities",
            "expected_file_type", "expected_max_streams", "expected_n_batch",
            "expected_n_ctx_seq", "expected_n_embd", "expected_n_layer",
            "expected_n_ubatch", "host", "io_timeout_ms", "launcher_component_id",
            "model_artifact", "model_id", "model_sha256", "port",
            "required_components", "route_epoch", "schema", "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "plan.cuda_monolithic_launch",
    )
    bundle = bundles["cuda_monolithic"]
    launcher = components[bundle["launcher_component_id"]]
    capture.exact(launch["schema"], "s39-cp0-r1-v24-cuda-monolithic-launch-v1", "launch.schema")
    capture.exact(launch["bundle_id"], "cuda_monolithic", "launch.bundle")
    capture.exact(launch["bundle_root"], roots["cuda_monolithic"], "launch.root")
    capture.exact(launch["endpoint"], "cuda", "launch.endpoint")
    capture.exact(launch["launcher_component_id"], bundle["launcher_component_id"], "launch.launcher")
    _Common.digest(launch["bundle_sha256"], "launch.bundle_sha256")
    port = _Common.integer(launch["port"], "launch.port", 1)
    capture.require(port <= 65535, "E_CUDA_MONOLITHIC_PORT")
    model_path = contract["model_geometry"][MODEL_ID]["cuda_model_path"]
    capture.exact(
        launch["command"],
        [
            launcher["path"], "-m", model_path, "--mode", "monov3", "--port",
            str(port), "--devices", "CUDA0", "--driver-batch", "8",
            "--driver-context", "512", "--driver-max-prefill", "8",
        ],
        "E_CUDA_MONOLITHIC_COMMAND",
    )
    capture.exact(
        launch["env"],
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HOME": "/home/zhihao",
            "LAYERSPLIT_MEMORY_CERT": "1",
            "LAYERSPLIT_MODEL_SHA256": token["model_sha256"],
            "LAYERSPLIT_PLACEMENT_CERT": "1",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": roots["cuda_monolithic"],
        },
        "E_CUDA_MONOLITHIC_ENV",
    )
    capture.exact(launch["cwd"], "/home/zhihao/llama.cpp-s40", "E_CUDA_MONOLITHIC_CWD")
    capture.exact(
        launch["allowed_system_roots"],
        ["/mnt/storage/s21_deps/cuda-13.2.1/lib/", "/usr/lib/x86_64-linux-gnu/"],
        "E_CUDA_MONOLITHIC_SYSTEM_ROOTS",
    )
    capture.exact(launch["host"], "127.0.0.1", "E_CUDA_MONOLITHIC_HOST")
    for key, expected in (
        ("expected_capabilities", 0x3F),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
        ("io_timeout_ms", 300000),
        ("shutdown_timeout_ms", 30000),
        ("startup_timeout_ms", 300000),
    ):
        capture.exact(launch[key], expected, f"launch.{key}")
    capture.exact(launch["model_id"], MODEL_ID, "launch.model_id")
    capture.exact(launch["model_sha256"], token["model_sha256"], "launch.model_sha256")
    capture.exact(
        launch["expected_file_type"],
        contract["cuda_monolithic_identity"]["expected_file_type"],
        "launch.file_type",
    )
    _Common.integer(launch["route_epoch"], "launch.route_epoch", 1)
    rows = launch["required_components"]
    capture.require(
        type(rows) is list and len(rows) == len(bundle["required_component_ids"]),
        "E_CUDA_MONOLITHIC_LAUNCH_COMPONENTS",
    )
    for index, (row, component_id) in enumerate(
        zip(rows, bundle["required_component_ids"], strict=True)
    ):
        field = f"launch.required_components[{index}]"
        _Common.exact_keys(row, {"component_id", "path", "sha256", "stat"}, field)
        capture.exact(row["component_id"], component_id, f"{field}.id")
        capture.exact(row["path"], components[component_id]["path"], f"{field}.path")
        capture.exact(row["sha256"], components[component_id]["sha256"], f"{field}.sha256")
        _Common.stat_record(row["stat"], f"{field}.stat")
    model = _Common.exact_keys(
        launch["model_artifact"],
        {"path", "sha256", "stat"},
        "launch.model_artifact",
    )
    capture.exact(model["path"], model_path, "launch.model.path")
    capture.exact(model["sha256"], token["model_sha256"], "launch.model.sha256")
    _Common.stat_record(model["stat"], "launch.model.stat")
    return launch


def _load_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_raw = _read_canonical(contract_path, "contract")
    candidate, candidate_raw = _read_canonical(candidate_path, "candidate")
    capture.exact(hashlib.sha256(contract_raw).hexdigest(), CONTRACT_SHA256, "contract.sha256")
    capture.exact(hashlib.sha256(candidate_raw).hexdigest(), CANDIDATE_SHA256, "candidate.sha256")
    capture.exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    capture.exact(candidate.get("schema"), "s39-cp0-r1-candidate-v1", "candidate.schema")
    capture.exact(contract["phase_protocol"]["phase"], PHASE, "contract.phase")
    capture.exact(contract["candidate_lock"]["sha256"], CANDIDATE_SHA256, "contract.candidate")
    model = [value for value in candidate["models"] if value.get("slot") == "A"]
    capture.require(len(model) == 1, "E_CANDIDATE_A")
    capture.exact(model[0]["model_id"], MODEL_ID, "candidate.model")
    return contract, contract_raw, candidate, candidate_raw


def _validate_runtime_plan(
    plan: dict[str, Any],
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> dict[str, Any]:
    _Common.exact_keys(
        plan,
        {
            "bundle_roots",
            "bundles",
            "candidate_sha256",
            "capture_entrypoints",
            "components",
            "contract_sha256",
            "cuda_monolithic_launch",
            "model_id",
            "phase",
            "schema",
            "token_history",
        },
        "runtime_plan",
    )
    capture.exact(plan["schema"], "s39-cp0-r1-runtime-bundle-plan-v2.4", "plan.schema")
    capture.exact(plan["phase"], PHASE, "plan.phase")
    capture.exact(plan["model_id"], MODEL_ID, "plan.model")
    capture.exact(plan["contract_sha256"], hashlib.sha256(contract_raw).hexdigest(), "plan.contract")
    capture.exact(plan["candidate_sha256"], hashlib.sha256(candidate_raw).hexdigest(), "plan.candidate")
    roots = _Common.exact_keys(plan["bundle_roots"], set(REQUIRED_BUNDLES), "plan.roots")
    root_paths = {}
    for bundle_id, root in roots.items():
        root_paths[bundle_id] = Path(_Common.absolute_path(root, f"plan.roots.{bundle_id}"))
    for left_id, left in root_paths.items():
        for right_id, right in root_paths.items():
            if left_id != right_id:
                capture.require(
                    not left.is_relative_to(right),
                    f"E_RUNTIME_ROOT_OVERLAP: {left_id}:{right_id}",
                )
    capture.require(type(plan["components"]) is list and bool(plan["components"]), "E_RUNTIME_COMPONENTS")
    components = {}
    locations = set()
    previous = None
    for index, value in enumerate(plan["components"]):
        field = f"plan.components[{index}]"
        _Common.exact_keys(
            value,
            {"bundle_id", "bytes", "component_id", "endpoint", "path", "role", "sha256"},
            field,
        )
        component_id = _Common.text(value["component_id"], f"{field}.id")
        capture.require(component_id not in components, f"E_COMPONENT_REUSE: {component_id}")
        if previous is not None:
            capture.require(previous < component_id, "E_COMPONENT_ORDER")
        previous = component_id
        capture.require(value["bundle_id"] in REQUIRED_BUNDLES, f"E_COMPONENT_BUNDLE: {field}")
        capture.exact(value["endpoint"], REQUIRED_BUNDLES[value["bundle_id"]][0], f"{field}.endpoint")
        _Common.integer(value["bytes"], f"{field}.bytes", 1)
        _Common.digest(value["sha256"], f"{field}.sha256")
        path = Path(_Common.absolute_path(value["path"], f"{field}.path"))
        capture.require(path.is_relative_to(root_paths[value["bundle_id"]]), f"E_RUNTIME_ROOT: {field}")
        location = (value["endpoint"], str(path))
        capture.require(location not in locations, f"E_RUNTIME_PATH_REUSE: {field}")
        locations.add(location)
        capture.require(value["role"] in {"backend_library", "executable", "shared_library"}, f"E_COMPONENT_ROLE: {field}")
        components[component_id] = value
    bundles = {}
    referenced = set()
    capture.require(
        type(plan["bundles"]) is list
        and len(plan["bundles"]) == len(REQUIRED_BUNDLES),
        "E_BUNDLES",
    )
    previous_bundle = None
    for index, value in enumerate(plan["bundles"]):
        field = f"plan.bundles[{index}]"
        _Common.exact_keys(
            value,
            {"bundle_id", "endpoint", "launcher_component_id", "process_role", "required_component_ids"},
            field,
        )
        bundle_id = value["bundle_id"]
        capture.require(bundle_id in REQUIRED_BUNDLES and bundle_id not in bundles, f"E_BUNDLE: {field}")
        if previous_bundle is not None:
            capture.require(previous_bundle < bundle_id, "E_BUNDLE_ORDER")
        previous_bundle = bundle_id
        endpoint, role = REQUIRED_BUNDLES[bundle_id]
        capture.exact(value["endpoint"], endpoint, f"{field}.endpoint")
        capture.exact(value["process_role"], role, f"{field}.role")
        required = value["required_component_ids"]
        capture.require(
            type(required) is list
            and bool(required)
            and required == sorted(set(required))
            and all(component_id in components for component_id in required),
            f"E_BUNDLE_COMPONENTS: {field}",
        )
        capture.require(value["launcher_component_id"] in required, f"E_BUNDLE_LAUNCHER: {field}")
        capture.exact(
            components[value["launcher_component_id"]]["role"],
            "executable",
            f"E_BUNDLE_LAUNCHER_ROLE: {field}",
        )
        for component_id in required:
            capture.exact(components[component_id]["bundle_id"], bundle_id, f"E_BUNDLE_OWNER: {component_id}")
        referenced.update(required)
        bundles[bundle_id] = value
    capture.exact(set(bundles), set(REQUIRED_BUNDLES), "plan.bundle_ids")
    captures = {}
    used = set()
    capture.require(
        type(plan["capture_entrypoints"]) is list
        and len(plan["capture_entrypoints"]) == len(CAPTURE_KINDS),
        "E_CAPTURE_ENTRYPOINTS",
    )
    previous_kind = None
    for value in plan["capture_entrypoints"]:
        _Common.exact_keys(
            value,
            {"component_id", "execution_mode", "kind", "nested_capture_entrypoint_component_ids"},
            "plan.capture",
        )
        kind = value["kind"]
        capture.require(kind in CAPTURE_KINDS and kind not in captures, "E_CAPTURE_KIND")
        if previous_kind is not None:
            capture.require(previous_kind < kind, "E_CAPTURE_ORDER")
        previous_kind = kind
        capture.exact(value["execution_mode"], "SELF_CONTAINED_PHYSICAL_CAPTURE", f"capture.{kind}.mode")
        component_id = value["component_id"]
        capture.require(component_id in components and component_id not in referenced and component_id not in used, f"E_CAPTURE_COMPONENT: {kind}")
        capture.exact(components[component_id]["role"], "executable", f"capture.{kind}.role")
        used.add(component_id)
        captures[kind] = value
    capture.exact(set(captures), CAPTURE_KINDS, "plan.capture_kinds")
    capture.exact(
        captures["artifact_root"]["nested_capture_entrypoint_component_ids"],
        [],
        "capture.artifact_root.nested",
    )
    capture.exact(
        captures["fast_fresh_readiness"]["nested_capture_entrypoint_component_ids"],
        [],
        "capture.fast_fresh_readiness.nested",
    )
    capture.exact(
        captures["cuda_monolithic"]["nested_capture_entrypoint_component_ids"],
        [bundles["cuda_monolithic"]["launcher_component_id"]],
        "capture.cuda_monolithic.nested",
    )
    capture.exact(
        captures["joint_phone_cuda"]["nested_capture_entrypoint_component_ids"],
        sorted(
            bundles[bundle_id]["launcher_component_id"]
            for bundle_id in (
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            )
        ),
        "capture.joint_phone_cuda.nested",
    )
    for kind in ("cuda_monolithic", "joint_phone_cuda"):
        source = contract["producer_requirements"]["source_programs"][kind]
        component = components[captures[kind]["component_id"]]
        capture.exact(component["bytes"], source["bytes"], f"capture.{kind}.bytes")
        capture.exact(component["sha256"], source["sha256"], f"capture.{kind}.sha256")
    capture.exact(referenced | used, set(components), "plan.component_closure")
    own = components[captures["artifact_root"]["component_id"]]
    own_raw = Path(__file__).read_bytes()
    capture.exact(own["bytes"], len(own_raw), "E_CAPTURE_SELF_BYTES")
    capture.exact(own["sha256"], hashlib.sha256(own_raw).hexdigest(), "E_CAPTURE_SELF_SHA256")
    token = _Common.exact_keys(
        plan["token_history"],
        {
            "artifact_path",
            "batch",
            "continuation_tokens_per_request",
            "corpus_sha256",
            "decode_calls_after_prefill",
            "mechanics_item_indices",
            "model_sha256",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "quality_group_count",
            "quality_group_size",
            "quality_items",
            "tokenizer_component_id",
            "tokenizer_plan_bytes",
            "tokenizer_plan_path",
            "tokenizer_plan_sha256",
        },
        "plan.token_history",
    )
    protocol = contract["token_history_protocol"]
    for key in (
        "batch",
        "continuation_tokens_per_request",
        "decode_calls_after_prefill",
        "mechanics_item_indices",
        "n_batch",
        "n_ctx_seq",
        "n_ubatch",
        "prefill_chunking",
        "prefill_row_order",
        "quality_group_count",
        "quality_group_size",
        "quality_items",
    ):
        capture.exact(token[key], protocol[key], f"plan.token_history.{key}")
    capture.exact(token["corpus_sha256"], contract["quality_corpus"]["sha256"], "plan.token_history.corpus")
    for key in ("artifact_path", "tokenizer_plan_path"):
        _Common.absolute_path(token[key], f"plan.token_history.{key}")
    capture.require(token["tokenizer_component_id"] in components, "E_TOKENIZER_COMPONENT")
    capture.exact(
        components[token["tokenizer_component_id"]]["endpoint"],
        "cuda",
        "E_TOKENIZER_ENDPOINT",
    )
    _Common.integer(token["tokenizer_plan_bytes"], "plan.tokenizer.bytes", 1)
    _Common.digest(token["tokenizer_plan_sha256"], "plan.tokenizer.sha256")
    cuda_launch = _validate_cuda_launch(
        plan["cuda_monolithic_launch"],
        contract,
        roots,
        bundles,
        components,
        token,
    )
    return {
        "bundles": bundles,
        "captures": captures,
        "components": components,
        "cuda_monolithic_launch": cuda_launch,
        "roots": roots,
        "token_history": token,
    }


def _expected_components(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    plan_derived: dict[str, Any],
    history_raw: bytes,
    tokenizer_plan_raw: bytes,
) -> dict[str, dict[str, Any]]:
    model = next(value for value in candidate["models"] if value["slot"] == "A")
    geometry = contract["model_geometry"][MODEL_ID]
    expected = {
        "model.cuda": {
            "bytes": model["artifact"]["bytes"],
            "endpoint": "cuda",
            "kind": "model_weight",
            "path": geometry["cuda_model_path"],
            "sha256": model["artifact"]["sha256"],
        },
        "model.op12_shard": {
            "bytes": geometry["known_shards"]["op12"]["bytes"],
            "endpoint": "op12",
            "kind": "model_shard",
            "path": geometry["known_shards"]["op12"]["path"],
            "sha256": geometry["known_shards"]["op12"]["sha256"],
        },
        "model.op15_shard": {
            "bytes": geometry["known_shards"]["op15"]["bytes"],
            "endpoint": "op15",
            "kind": "model_shard",
            "path": geometry["known_shards"]["op15"]["path"],
            "sha256": geometry["known_shards"]["op15"]["sha256"],
        },
    }
    expected["token_history.mmlu64"] = {
        "bytes": len(history_raw),
        "endpoint": "cuda",
        "kind": "token_history",
        "path": plan_derived["token_history"]["artifact_path"],
        "sha256": hashlib.sha256(history_raw).hexdigest(),
    }
    expected["tokenizer.plan"] = {
        "bytes": len(tokenizer_plan_raw),
        "endpoint": "cuda",
        "kind": "tokenizer_plan",
        "path": plan_derived["token_history"]["tokenizer_plan_path"],
        "sha256": hashlib.sha256(tokenizer_plan_raw).hexdigest(),
    }
    for component_id, value in plan_derived["components"].items():
        expected[component_id] = {
            "bytes": value["bytes"],
            "endpoint": value["endpoint"],
            "kind": "runtime_component",
            "path": value["path"],
            "sha256": value["sha256"],
        }
    return expected


def _inventories(
    runner: capture.Runner,
    authority: types.ModuleType,
    plan_derived: dict[str, Any],
    serials: dict[str, str],
    timeout: float,
) -> list[dict[str, Any]]:
    del runner, authority, serials, timeout
    result = []
    for bundle_id in sorted(plan_derived["bundles"]):
        bundle = plan_derived["bundles"][bundle_id]
        expected_paths = sorted(
            plan_derived["components"][component_id]["path"]
            for component_id in bundle["required_component_ids"]
        )
        result.append(
            {
                "bundle_id": bundle_id,
                "endpoint": bundle["endpoint"],
                "paths": expected_paths,
                "root": plan_derived["roots"][bundle_id],
            }
        )
    return result


def _bundle_digest(
    bundle_id: str,
    plan_derived: dict[str, Any],
    components: dict[str, dict[str, Any]],
) -> str:
    bundle = plan_derived["bundles"][bundle_id]
    identity = {
        "bundle_id": bundle_id,
        "components": [
            {
                "component_id": component_id,
                "path": components[component_id]["path"],
                "sha256": components[component_id]["sha256"],
                "stat": components[component_id]["stat"],
            }
            for component_id in bundle["required_component_ids"]
        ],
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "process_role": bundle["process_role"],
        "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _validate_output(
    value: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    inventories: list[dict[str, Any]],
    plan_derived: dict[str, Any],
) -> None:
    _Common.exact_keys(
        value,
        {
            "candidate_sha256",
            "completed_ns",
            "components",
            "contract_sha256",
            "inventories",
            "model_id",
            "phase_scope",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "artifact_root",
    )
    capture.exact(value["schema"], SCHEMA, "root.schema")
    capture.require(value["started_ns"] < value["completed_ns"], "E_ROOT_INTERVAL")
    capture.exact(value["inventories"], inventories, "root.inventories")
    by_id = {}
    previous = None
    for row in value["components"]:
        component_id = row["component_id"]
        capture.require(component_id in expected and component_id not in by_id, "E_ROOT_COMPONENT")
        if previous is not None:
            capture.require(previous < component_id, "E_ROOT_COMPONENT_ORDER")
        previous = component_id
        for key, expected_value in expected[component_id].items():
            capture.exact(row[key], expected_value, f"root.{component_id}.{key}")
        _Common.stat_record(row["stat"], f"root.{component_id}.stat")
        capture.require(stat.S_ISREG(row["stat"]["mode"]), f"E_ROOT_REGULAR: {component_id}")
        capture.exact(row["stat"]["size"], row["bytes"], f"root.{component_id}.size")
        by_id[component_id] = row
    capture.exact(set(by_id), set(expected), "root.component_ids")
    launch = plan_derived["cuda_monolithic_launch"]
    for row in launch["required_components"]:
        capture.exact(row["stat"], by_id[row["component_id"]]["stat"], "E_LAUNCH_STAT")
    capture.exact(launch["model_artifact"]["stat"], by_id["model.cuda"]["stat"], "E_MODEL_STAT")
    capture.exact(
        launch["bundle_sha256"],
        _bundle_digest("cuda_monolithic", plan_derived, by_id),
        "E_CUDA_LAUNCH_BUNDLE_DIGEST",
    )


def capture_artifact_root(
    *,
    output: Path,
    contract_path: Path,
    candidate_path: Path,
    runtime_plan_path: Path,
    history_path: Path,
    tokenizer_plan_path: Path,
    cuda_ssh_target: str,
    phone_adb_port: int,
    confirmation: str,
    runner: capture.Runner | None = None,
    now_ns: Callable[[], int] = capture.clock_ns,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    capture.exact(
        confirmation,
        capture.CONFIRM_ARTIFACT,
        "confirmation",
    )
    capture.exact(cuda_ssh_target, capture.CUDA_SSH_TARGET, "cuda_ssh_target")
    capture.exact(phone_adb_port, capture.PHONE_ADB_PORT, "phone_adb_port")
    capture.integer(timeout_seconds, "timeout_seconds", 1)
    capture.require(timeout_seconds <= 7200, "E_TIMEOUT_RANGE")
    capture.require(output.is_absolute() and not output.exists(), "E_OUTPUT")

    contract, contract_raw, candidate, candidate_raw = _load_inputs(
        contract_path,
        candidate_path,
    )
    plan, plan_raw = _read_canonical(runtime_plan_path, "runtime_plan")
    plan_derived = _validate_runtime_plan(
        plan,
        contract,
        contract_raw,
        candidate_raw,
    )
    tokenizer_plan, tokenizer_plan_raw = _read_canonical(tokenizer_plan_path, "tokenizer_plan")
    history, history_raw = _read_canonical(history_path, "token_history")
    capture.exact(tokenizer_plan.get("schema"), "s39-cp0-r1-a-only-tokenizer-plan-v2", "tokenizer.schema")
    capture.exact(history.get("schema"), "s39-cp0-r1-token-history-v2.4", "history.schema")
    capture.exact(history.get("model_id"), MODEL_ID, "history.model")
    token_spec = plan_derived["token_history"]
    capture.exact(len(tokenizer_plan_raw), token_spec["tokenizer_plan_bytes"], "tokenizer.bytes")
    capture.exact(hashlib.sha256(tokenizer_plan_raw).hexdigest(), token_spec["tokenizer_plan_sha256"], "tokenizer.sha256")
    capture.exact(tokenizer_plan["component_id"], token_spec["tokenizer_component_id"], "tokenizer.component")
    capture.exact(history["candidate_sha256"], CANDIDATE_SHA256, "history.candidate")
    capture.exact(history["corpus_sha256"], contract["quality_corpus"]["sha256"], "history.corpus")
    expected = _expected_components(
        contract,
        candidate,
        plan_derived,
        history_raw,
        tokenizer_plan_raw,
    )
    serials = {
        endpoint: contract["devices"][endpoint]["serial"]
        for endpoint in ("op12", "op15")
    }
    runner = runner or capture.SubprocessRunner()
    started_ns = capture.integer(now_ns(), "started_ns", 1)
    records = []
    for component_id in sorted(expected):
        spec = expected[component_id]
        collected = capture.collect_artifact(
            runner,
            AUTHORITY_SHIM,
            spec["endpoint"],
            spec["path"],
            serials,
            include_digest=True,
            timeout=timeout_seconds,
        )
        capture.exact(
            collected["stat"]["size"],
            spec["bytes"],
            f"E_BYTES: {component_id}",
        )
        capture.exact(
            collected["sha256"],
            spec["sha256"],
            f"E_SHA256: {component_id}",
        )
        records.append(
            {
                **spec,
                "component_id": component_id,
                "stat": collected["stat"],
            }
        )
    inventories = _inventories(
        runner,
        AUTHORITY_SHIM,
        plan_derived,
        serials,
        timeout_seconds,
    )
    completed_ns = capture.integer(now_ns(), "completed_ns", 1)
    capture.require(started_ns < completed_ns, "E_ARTIFACT_ROOT_INTERVAL")
    result = {
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "completed_ns": completed_ns,
        "components": records,
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "inventories": inventories,
        "model_id": MODEL_ID,
        "phase_scope": "PRE_REBOOT_OUTSIDE_PHASE",
        "runtime_bundle_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "schema": SCHEMA,
        "started_ns": started_ns,
    }
    _validate_output(result, expected, inventories, plan_derived)
    raw = _canonical_bytes(result)
    capture.durable_write_new(output, raw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--tokenizer-plan", type=Path, required=True)
    parser.add_argument("--cuda-ssh-target", required=True)
    parser.add_argument("--phone-adb-port", type=int, required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    try:
        value = capture_artifact_root(
            output=args.output,
            contract_path=args.contract,
            candidate_path=args.candidate,
            runtime_plan_path=args.runtime_plan,
            history_path=args.history,
            tokenizer_plan_path=args.tokenizer_plan,
            cuda_ssh_target=args.cuda_ssh_target,
            phone_adb_port=args.phone_adb_port,
            confirmation=args.confirm,
            timeout_seconds=args.timeout_seconds,
        )
        del value
        return 0
    except Exception as error:
        print(
            f"V24_ARTIFACT_ROOT_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
