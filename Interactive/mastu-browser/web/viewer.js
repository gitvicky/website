/* Dependency-free WebGL2 renderer. Coordinates and displacement are in metres. */
(function (root) {
  'use strict';
  const VERTEX = `#version 300 es
    precision highp float;
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    layout(location=2) in vec3 displacement;
    uniform mat4 viewProjection;
    uniform float amplification;
    uniform float colourMaximum;
    uniform int quantity;
    out vec3 vNormal;
    out float vScalar;
    void main() {
      gl_Position = viewProjection * vec4(position + amplification * displacement, 1.0);
      vNormal = normal;
      float value = quantity == 0 ? length(displacement) : displacement[quantity - 1];
      vScalar = quantity == 0 ? value * 1000.0 / colourMaximum : 0.5 + 0.5 * value * 1000.0 / colourMaximum;
    }`;
  const FRAGMENT = `#version 300 es
    precision highp float;
    in vec3 vNormal;
    in float vScalar;
    out vec4 colour;
    vec3 ramp(float v) {
      vec3 a=vec3(.230,.299,.754), b=vec3(.554,.690,.996), c=vec3(.865,.865,.865), d=vec3(.956,.598,.478), e=vec3(.706,.016,.150);
      float t=clamp(v,0.0,1.0)*4.0;
      return t<1.0 ? mix(a,b,t) : t<2.0 ? mix(b,c,t-1.0) : t<3.0 ? mix(c,d,t-2.0) : mix(d,e,t-3.0);
    }
    void main() {
      vec3 n=normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
      float light=.47+.36*max(dot(n,normalize(vec3(1.0,-.4,1.5))),0.0)+.17*max(dot(n,normalize(vec3(-1.0,1.0,.4))),0.0);
      colour=vec4(ramp(vScalar)*light,1.0);
    }`;
  const sub=(a,b)=>a.map((v,i)=>v-b[i]);
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);};
  function multiply(a,b) {
    const out=new Float32Array(16);
    for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)out[col*4+row]+=a[k*4+row]*b[col*4+k];
    return out;
  }
  class MastuViewer {
    constructor(canvas, positions, normals, triangles, colourMaximum) {
      this.canvas=canvas;
      const gl=canvas.getContext('webgl2',{alpha:true,antialias:true});
      if(!gl)throw Error('This interactive needs WebGL2. Try a current browser with hardware acceleration enabled.');
      this.gl=gl;
      const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
      const program=gl.createProgram();
      gl.attachShader(program,shader(gl.VERTEX_SHADER,VERTEX));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,FRAGMENT));gl.linkProgram(program);
      if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);this.program=program;
      this.uniforms=Object.fromEntries(['viewProjection','amplification','colourMaximum','quantity'].map(k=>[k,gl.getUniformLocation(program,k)]));
      this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);
      const attribute=(index,values,usage)=>{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,values,usage);gl.enableVertexAttribArray(index);gl.vertexAttribPointer(index,3,gl.FLOAT,false,0,0);return b;};
      attribute(0,positions,gl.STATIC_DRAW);attribute(1,normals,gl.STATIC_DRAW);
      this.fieldBuffer=attribute(2,new Float32Array(positions.length),gl.DYNAMIC_DRAW);
      const indices=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,triangles,gl.STATIC_DRAW);
      this.count=triangles.length;this.colourMaximum=colourMaximum;this.amplification=0;this.quantity=0;
      const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
      for(let i=0;i<positions.length;i++) {const j=i%3;lo[j]=Math.min(lo[j],positions[i]);hi[j]=Math.max(hi[j],positions[i]);}
      this.target=lo.map((v,i)=>(v+hi[i])/2);this.radius=Math.hypot(...sub(hi,lo))/2;
      this.reset();this.bindControls();
      this.resizeObserver=new ResizeObserver(()=>this.draw());this.resizeObserver.observe(canvas);
      canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();canvas.dispatchEvent(new CustomEvent('viewer-error',{detail:'The graphics context was lost. Reload this page to restore the viewer.'}));});
      gl.enable(gl.DEPTH_TEST);gl.clearColor(0,0,0,0);this.draw();
    }
    reset(){this.azimuth=Math.PI/4;this.elevation=.16;this.distance=this.radius*2.7;this.draw();}
    bindControls(){
      let previous=null;
      this.canvas.addEventListener('pointerdown',e=>{previous=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);});
      this.canvas.addEventListener('pointermove',e=>{if(!previous)return;this.azimuth-=(e.clientX-previous[0])*.007;this.elevation=Math.max(-1.35,Math.min(1.35,this.elevation+(e.clientY-previous[1])*.007));previous=[e.clientX,e.clientY];this.draw();});
      const stop=()=>{previous=null;};
      this.canvas.addEventListener('pointerup',stop);this.canvas.addEventListener('pointercancel',stop);this.canvas.addEventListener('lostpointercapture',stop);
      this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(this.radius*.7,Math.min(this.radius*7,this.distance*Math.exp(e.deltaY*.001)));this.draw();},{passive:false});
      this.canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','Home'].includes(e.key)){e.preventDefault();if(e.key==='Home')this.reset();if(e.key==='ArrowLeft')this.azimuth-=.1;if(e.key==='ArrowRight')this.azimuth+=.1;if(e.key==='ArrowUp')this.elevation=Math.min(1.35,this.elevation+.1);if(e.key==='ArrowDown')this.elevation=Math.max(-1.35,this.elevation-.1);if(e.key==='+')this.distance=Math.max(this.radius*.7,this.distance*.9);if(e.key==='-')this.distance=Math.min(this.radius*7,this.distance*1.1);this.draw();}});
    }
    update(field){const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.fieldBuffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,field);this.draw();}
    draw(){
      if(!this.gl||!this.radius)return;
      const gl=this.gl,c=this.canvas,dpr=Math.min(devicePixelRatio||1,2),width=Math.max(1,Math.round(c.clientWidth*dpr)),height=Math.max(1,Math.round(c.clientHeight*dpr));
      if(c.width!==width||c.height!==height){c.width=width;c.height=height;}
      gl.viewport(0,0,width,height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);gl.bindVertexArray(this.vao);
      const ce=Math.cos(this.elevation),eye=[this.target[0]+this.distance*ce*Math.cos(this.azimuth),this.target[1]+this.distance*ce*Math.sin(this.azimuth),this.target[2]+this.distance*Math.sin(this.elevation)];
      const z=unit(sub(eye,this.target)),x=unit(cross([0,0,1],z)),y=cross(z,x);
      const view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
      const f=1/Math.tan(Math.PI/8),near=this.radius*.01,far=this.radius*30;
      const proj=new Float32Array([f/(width/height),0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
      gl.uniformMatrix4fv(this.uniforms.viewProjection,false,multiply(proj,view));gl.uniform1f(this.uniforms.amplification,this.amplification);gl.uniform1f(this.uniforms.colourMaximum,this.colourMaximum);gl.uniform1i(this.uniforms.quantity,this.quantity);
      gl.drawElements(gl.TRIANGLES,this.count,gl.UNSIGNED_INT,0);
    }
  }
  root.MastuViewer=MastuViewer;
})(globalThis);
