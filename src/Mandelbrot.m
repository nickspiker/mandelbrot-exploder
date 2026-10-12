%Mandelbrot rendering algorithms utilizing interior and exterior slope finding steps copyright 2020 Nick Spiker nick@nickspiker.com
maxn=32;
mns=32;%max newton steps
eps=2^-53;%epsilon squared2^-53 is minimum precision
insidep=false;
%orbitcount=1;
crp=1.8;
cip=0.3;
zr=crp;
zi=cip;
ap=maxn;
drp=0;
dip=0;
dsp=0;
erp=0;
eip=0;
esp=0;
srp=1;
sip=0;
osr=1;
osi=0;
%orbit=0;
zsmallest=realmax;
for n=1:maxn
    
    srpt=2.*(srp.*zr-sip.*zi);
    sip=2.*(srp.*zi+sip.*zr);
    srp=srpt+1;
    
    
    zrs=zr.*zr;
    zis=zi.*zi;
    zs=zrs+zis;
    ns=n.*n;
    if zs>4
        llzs=log(log(zs));
        taper=(llzs-0.326634259978281)./6.238324625039508;
        taper=1-taper.*taper;
        taper=taper.*taper;
        taper=taper.*taper.*taper.*ns;
        zsq=taper./sqrt(zs);
        esp=esp+taper;
        erp=erp+zr.*zsq;
        eip=eip+zi.*zsq;
        
        zsi=ns./zs;
        dsp=dsp+zsi;
        drp=drp+zr.*zsi;
        dip=dip+zi.*zsi;
        
        if zs>1.340780792994260e+154
            ap=n-(llzs-5.174750624576761).*1.442695040888963;
            break
        end
        
        durt=2.*(zr.*osr-zi.*osi)+1;
        osi=2.*(osr.*zi+osi.*zr);
        osr=durt;
        
        zi=2.*zr.*zi+cip;
        zr=zrs-zis+crp;
    else
        taper=ns;
        zsq=taper./sqrt(zs);
        esp=esp+taper;
        erp=erp+zr.*zsq;
        eip=eip+zi.*zsq;
        zsi=ns./zs;
        dsp=dsp+zsi;
        drp=drp+zr.*zsi;
        dip=dip+zi.*zsi;
        if zsmallest>zs
            adcs=zs./zsmallest;
            zsmallest=zs;
            if adcs<.25
                period=n;
                wr=zr;
                wi=zi;
                for l=1:mns
                    ur=wr;
                    ui=wi;
                    dur=1;
                    dui=0;
                    for m=1:period
                        
                        %du=2*du*u
                        durt=2.*(dur.*ur-dui.*ui);
                        dui=2.*(dur.*ui+dui.*ur);
                        dur=durt;
                        
                        %u=u*u+c
                        uis=ui.*ui;
                        ui=2.*ur.*ui+cip;
                        ur=ur.*ur-uis+crp;
                        
                    end
                    
                    %nw=w-(u-w)/(du-1)
                    nwr=ur-wr;
                    nwi=ui-wi;
                    duro=dur-1;
                    durt=duro.*duro+dui.*dui;
                    nwrt=(nwr.*duro+nwi.*dui)./durt;
                    nwi=(nwi.*duro-nwr.*dui)./durt;
                    nwr=nwrt;
                    ns=nwr.*nwr+nwi.*nwi;
                    wr=wr-nwr;
                    wi=wi-nwi;
                    
                    if ns<eps
                        if dur.*dur+dui.*dui<1
                            drp=dur;
                            dip=dui;
                            erp=wr;
                            eip=wi;
                            insidep=true;
                            ap=m;
                            break
                        end
                    end
                end
                if insidep
                    break
                end
            end
        end
        
        durt=2.*(zr.*osr-zi.*osi)+1;
        osi=2.*(osr.*zi+osi.*zr);
        osr=durt;
        
        zi=2.*zr.*zi+cip;
        zr=zrs-zis+crp;
        
    end
    if insidep
        break
    end
end

if insidep
    inside(p)=true;
    dr(p)=drp;
    di(p)=dip;
    er(p)=erp;
    ei(p)=eip;
    
    cca=crp+cip.*i;
    ccc=cca;
    for l=1:mns
        zzz=ccc;
        ddd=1;
        for z=2:period
            ddd=2.*ddd.*zzz+1;
            zzz=zzz.*zzz+ccc;
        end
        ddd=ccc-zzz./ddd+2^-256.*i;
        if ddd==ccc
            ccc=ddd;
            break
        end
        ccc=ddd;
    end
    slope(p)=ccc-cca;
else
    dsp=1./dsp;
    esp=1./esp;
    drp=drp.*dsp;
    dip=dip.*dsp;
    erp=erp.*esp;
    eip=eip.*esp;
    if n<maxn
        slopep=((zr-zi.*i)*log(zr^2+zi^2))/(osr-osi.*i);
    end
end
a=ap;
