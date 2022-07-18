// Fenced code from: https://github.com/Twinklebear/webgl-volume-raycaster

import React, { Component } from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { State_t } from '../../app/model'
import { fetch_data_cache } from '../../api'

import { selectors } from '../'
import { apply_green_LUT } from '../helpers'

import { mat4, vec3 } from '../webgl/gl-matrix'
import { ArcballCamera, Shader, Controller } from '../webgl/webgl-util'
import { fragShader, vertShader } from '../webgl/shader-srcs'

const cubeStrip = [
  1, 1, 0,
  0, 1, 0,
  1, 1, 1,
  0, 1, 1,
  0, 0, 1,
  0, 1, 0,
  0, 0, 0,
  1, 1, 0,
  1, 0, 0,
  1, 1, 1,
  1, 0, 1,
  0, 0, 1,
  1, 0, 0,
  0, 0, 0
];

const mapStateToProps = (state: State_t) => {
  return {
    shape_x: selectors.get_shape_x(state),
    shape_y: selectors.get_shape_y(state),
    shape_z: selectors.get_shape_z(state),
    scale_x: selectors.get_scale_x(state),
    scale_y: selectors.get_scale_y(state),
    scale_z: selectors.get_scale_z(state),
    height: (selectors.get_shape_y(state) * selectors.get_scale_y(state)
      + selectors.get_shape_z(state) * selectors.get_scale_z(state)),
    width: (selectors.get_shape_x(state) * selectors.get_scale_x(state)
      + selectors.get_shape_z(state) * selectors.get_scale_z(state)),
    data_url: selectors.get_vol_url(state),
    x: selectors.get_x(state),
    y: selectors.get_y(state),
    z: selectors.get_z(state),
    LUT: selectors.get_LUT(state),
    gamma: selectors.get_gamma(state)
  }
}

const connector = connect(mapStateToProps)

type ReduxProps_t = ConnectedProps<typeof connector>
type VolumeView_props_t = ReduxProps_t & {
  left: number,
  top: number,
}

type VolumeView_state_t = {
  data: Uint8ClampedArray | null
}

class _VolumeView
  extends Component<VolumeView_props_t, VolumeView_state_t> {

  state: VolumeView_state_t
  abort_controller: AbortController

  // Begin from volume-raycaster.js

  canvas: HTMLCanvasElement | null = null
  gl: WebGL2RenderingContext | null = null
  shader: Shader | null = null
  volumeTexture: WebGLTexture | null = null
  proj: mat4 = mat4.create()
  camera = new ArcballCamera(vec3.create(), vec3.create(), vec3.create(), 2,
    [500, 500])
  projView = mat4.create()
  targetFrameTime = 50
  samplingRate = 1.0
  WIDTH = 640
  HEIGHT = 480

  static readonly defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 1.5)
  static readonly center = vec3.set(vec3.create(), 0.5, 0.5, 0.5)
  static readonly up = vec3.set(vec3.create(), 0.0, 1.0, 0.0)

  // End from volume-raycaster.js

  constructor(props: VolumeView_props_t) {
    super(props)
    this.abort_controller = new AbortController()
    this.state = {
      data: null,
    }

    // Begin from volume-raycaster.js

    this.WIDTH = props.width
    this.HEIGHT = props.height
    this.proj = mat4.perspective(mat4.create(), 60 * Math.PI / 180.0,
      this.WIDTH / this.HEIGHT, 0.1, 100)

    this.camera = new ArcballCamera(
      _VolumeView.defaultEye, _VolumeView.center, _VolumeView.up,
      2,
      [this.WIDTH, this.HEIGHT])

    // End from volume-raycaster.js
  }

  componentDidMount() {

    this.canvas = this.refs.canvas as HTMLCanvasElement
    this.gl = this.canvas.getContext("webgl2") as WebGL2RenderingContext

    const gl = this.gl

    const controller = new Controller(this.camera);
    controller.mousemove = function (prev: number[], cur: number[],
      evt: MouseEvent) {

      if (evt.buttons === 1) {
        this.camera.rotate(prev, cur);

      } else if (evt.buttons === 2) {
        this.camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
      }
    };
    controller.wheel = function (amt: number) { this.camera.zoom(amt); };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function (drag: number) {
      this.camera.pan(drag);
    }
    controller.registerForCanvas(this.canvas);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    var vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array(cubeStrip), gl.STATIC_DRAW);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    this.shader = new Shader(gl, vertShader, fragShader);
    this.shader.use(gl);

    // @ts-ignore Property 'volume' does not exist on type '{}'
    gl.uniform1i(this.shader.uniforms["volume"], 0);
    // @ts-ignore
    gl.uniform1i(this.shader.uniforms["colormap"], 1);
    // @ts-ignore
    gl.uniform1f(this.shader.uniforms["dt_scale"], this.samplingRate);

    // Setup required OpenGL state for drawing the back faces and
    // composting with the background color
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.FRONT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Load the default colormap and upload it, then draw the volume.
    const colormapImage = new Image();
    colormapImage.onload = ((gl, draw) => function () {
      var colormap = gl.createTexture();
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, colormap);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.SRGB8_ALPHA8, 180, 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
        gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);

      draw();
    })(gl, this.draw_to_canvas)
    colormapImage.src = "webgl_colormaps/samsel-linear-green.png";

    // End from volume-raycaster.js

    this.fetch_and_draw()

  }

  componentDidUpdate(prev_props: VolumeView_props_t) {
    if (this.props.data_url !== prev_props.data_url)
      this.fetch_and_draw()
    else if (this.props.LUT !== prev_props.LUT)
      this.draw_to_canvas()
  }

  fetch_and_draw = () => {

    this.abort_controller.abort()
    this.abort_controller = new AbortController()

    fetch_data_cache(this.props.data_url, this.abort_controller.signal)
      .then(data => this.setState({ data }))
      .then(this.draw_to_canvas)
  }

  draw_to_canvas = () => {

    if ((this.state.data === null) ||
      (this.state.data === undefined) ||
      (this.props.width === 0) ||
      (this.props.height === 0) ||
      (this.gl === null) ||
      (this.camera === null)) {

      return
    }

    const gl = this.gl
    const volDims = [
      this.props.shape_x,
      this.props.shape_y,
      this.props.shape_z
    ]

    const LUTdata = apply_green_LUT(this.state.data, this.props.LUT, this.props.gamma)

    const tex = gl.createTexture();

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, volDims[0], volDims[1], volDims[2]);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0,
      volDims[0], volDims[1], volDims[2],
      gl.RED, gl.UNSIGNED_BYTE, LUTdata);

    var longestAxis = Math.max(volDims[0], Math.max(volDims[1], volDims[2]));
    var volScale = [
      volDims[0] / longestAxis * this.props.scale_x,
      volDims[1] / longestAxis * this.props.scale_y,
      volDims[2] / longestAxis * this.props.scale_z,
    ]

    // @ts-ignore Property 'volume_dims' does not exist on type '{}'
    gl.uniform3iv(this.shader.uniforms["volume_dims"], volDims)
    // @ts-ignore Property 'volume_scale' does not exist on type '{}'
    gl.uniform3fv(this.shader.uniforms["volume_scale"], volScale)

    if (!this.volumeTexture) {
      this.volumeTexture = tex;
      setInterval(() => {

        // Save them some battery if they're not viewing the tab
        if (document.hidden) {
          return;
        }
        const startTime = new Date()
        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT)

        this.projView = mat4.mul(this.projView as mat4, this.proj,
          this.camera.camera);

        // @ts-ignore Property 'proj_view' does not exist on type '{}'
        gl.uniformMatrix4fv(this.shader.uniforms["proj_view"],
          false,
          this.projView)

        var eye = [this.camera.invCamera[12],
        this.camera.invCamera[13],
        this.camera.invCamera[14]]

        // @ts-ignore Property 'eye_pos' does not exist on type '{}'
        gl.uniform3fv(this.shader.uniforms["eye_pos"], eye);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, cubeStrip.length / 3);
        // Wait for rendering to actually finish
        gl.finish()

        const endTime = new Date();
        const renderTime = (endTime.getTime() - startTime.getTime())
        const targetSamplingRate = renderTime / this.targetFrameTime
        if (this.samplingRate < targetSamplingRate) {
          this.samplingRate = (
            0.8 * this.samplingRate +
            0.2 * targetSamplingRate)
          // @ts-ignore
          gl.uniform1f(this.shader.uniforms["dt_scale"], this.samplingRate);
        }

      }, this.targetFrameTime)
    }
    else {
      gl.deleteTexture(this.volumeTexture);
      this.volumeTexture = tex;
    }

  }

  render() {

    return (
      <div style={{
        position: "relative",
        left: this.props.left,
        top: this.props.top,
        right: "0px",
      }}>

        <canvas ref="canvas"
          width={this.props.width}
          height={this.props.height}
          style={{
            position: "absolute",
            left: "0px",
            top: "0px",
            zIndex: 1,
          }}>
        </canvas>

      </div>)
  }
}

export const VolumeView = connector(_VolumeView)
