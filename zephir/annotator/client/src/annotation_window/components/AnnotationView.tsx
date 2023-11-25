import React, { Component } from 'react'
import { connect, ConnectedProps } from 'react-redux'

import { State_t } from '../../app/model'
import { AppDispatch_t } from '../..'
import { fetch_data_cache } from '../../api'

import { selectors } from '../'
import { apply_LUT } from '../helpers'
import { saga_actions, click_payload_t } from '../sagas'
import { AnnotationsSVG } from '../../annotations'

import { selectors as annotations_selectors } from '../../annotations'

const mapStateToProps_X = (state: State_t) => {
  return {
    height: selectors.get_shape_y(state),
    width: selectors.get_shape_z(state),
    height_scale: selectors.get_scale_y(state),
    width_scale: selectors.get_scale_z(state),
    data_url: selectors.get_x_url(state),
    x: selectors.get_z(state),
    y: selectors.get_y(state),
    x_dim: "z",
    y_dim: "y",
    annotations: annotations_selectors.get_x_annotations(state),
    fill_circles: selectors.get_fill_circles(state),
    LUT: selectors.get_LUT(state),
    gamma: selectors.get_gamma(state)
  }
}

const mapStateToProps_Y = (state: State_t) => {
  return {
    height: selectors.get_shape_z(state),
    width: selectors.get_shape_x(state),
    height_scale: selectors.get_scale_z(state),
    width_scale: selectors.get_scale_x(state),
    data_url: selectors.get_y_url(state),
    x: selectors.get_x(state),
    y: selectors.get_z(state),
    x_dim: "x",
    y_dim: "z",
    annotations: annotations_selectors.get_y_annotations(state),
    fill_circles: selectors.get_fill_circles(state),
    LUT: selectors.get_LUT(state),
    gamma: selectors.get_gamma(state)
  }
}

const mapStateToProps_Z = (state: State_t) => {
  return {
    height: selectors.get_shape_y(state),
    width: selectors.get_shape_x(state),
    height_scale: selectors.get_scale_y(state),
    width_scale: selectors.get_scale_x(state),
    data_url: selectors.get_z_url(state),
    x: selectors.get_x(state),
    y: selectors.get_y(state),
    x_dim: "x",
    y_dim: "y",
    annotations: annotations_selectors.get_z_annotations(state),
    fill_circles: selectors.get_fill_circles(state),
    LUT: selectors.get_LUT(state),
    gamma: selectors.get_gamma(state)
  }
}

const mapDispatchToProps = (dispatch: AppDispatch_t) => {
  return {
    select: (x: number) => dispatch(saga_actions.set_selected_annotation(x)),
    deselect: () => dispatch(saga_actions.set_selected_annotation(null)),
    click: (x: click_payload_t) => dispatch(saga_actions.click(x)),
    // doubleclick: (x: click_payload_t) =>
    //   dispatch(saga_actions.doubleclick(x))
    // drag: (x: click_payload_t) => dispatch(saga_actions.drag(x))
  }
}

const connector_X = connect(mapStateToProps_X, mapDispatchToProps)
const connector_Y = connect(mapStateToProps_Y, mapDispatchToProps)
const connector_Z = connect(mapStateToProps_Z, mapDispatchToProps)

type ReduxProps_t = ConnectedProps<typeof connector_X>
type AnnotationView_props_t = ReduxProps_t & {
  left: number,
  top: number,
}

type AnnotationView_state_t = {
  data: Uint8ClampedArray | null
}

class AnnotationView
  extends Component<AnnotationView_props_t, AnnotationView_state_t> {

  state: AnnotationView_state_t
  abort_controller: AbortController

  constructor(props: AnnotationView_props_t) {
    super(props)
    this.abort_controller = new AbortController()
    this.state = {
      data: null,
    }
  }

  componentDidMount() {
    this.fetch_and_draw()
  }

  componentDidUpdate(prev_props: AnnotationView_props_t) {
    if (this.props.data_url !== prev_props.data_url)
      this.fetch_and_draw()
    else if (this.props.LUT !== prev_props.LUT)
      this.draw_to_canvas()
    else if (this.props.gamma !== prev_props.gamma)
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

    if ((this.state.data != null) &&
      (this.props.width > 0) &&
      (this.props.height > 0)) {

      const LUTdata = apply_LUT(this.state.data, this.props.LUT, this.props.gamma)

      const cvs = this.refs.canvas as HTMLCanvasElement
      const ctx = cvs.getContext("2d")

      const iData = new ImageData(
        LUTdata,
        this.props.width,
        this.props.height);

      ctx?.putImageData(iData, 0, 0)

    }
  }

  make_click_handler = (click_fn: any) => (event: React.MouseEvent) => {
    const target = event.target as HTMLElement;
    const dim = target.getBoundingClientRect();
    const x_pixels = event.clientX - dim.left;
    const y_pixels = event.clientY - dim.top;

    const svg_width = this.props.width * this.props.width_scale
    const svg_height = this.props.height * this.props.height_scale

    const x_local = x_pixels / svg_width
    const y_local = y_pixels / svg_height

    click_fn({
      [this.props.x_dim]: x_local,
      [this.props.y_dim]: y_local
    })

  }

  handle_click = this.make_click_handler(this.props.click)
  // handle_doubleclick = this.make_click_handler(this.props.doubleclick)

  render() {

    const svg_width = this.props.width * this.props.width_scale
    const svg_height = this.props.height * this.props.height_scale

    const hline =
      <line
        x1={0}
        y1={this.props.y * svg_height}
        x2={svg_width}
        y2={this.props.y * svg_height}
        strokeWidth="0.4" stroke="grey" />
    const vline =
      <line
        x1={this.props.x * svg_width}
        y1={0}
        x2={this.props.x * svg_width}
        y2={svg_height}
        strokeWidth="0.4" stroke="grey" />

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
            width: this.props.width * this.props.width_scale,
            height: this.props.height * this.props.height_scale
          }}>
        </canvas>

        <svg ref="svg"
          width={this.props.width * this.props.width_scale}
          height={this.props.height * this.props.height_scale}
          onClick={this.handle_click}
          // onContextMenu={this.handle_doubleclick}
          // onDoubleClick={this.handle_doubleclick}
          style={{
            position: "absolute",
            left: "0px",
            top: "0px",
            zIndex: 2,
          }}>
          <AnnotationsSVG
            annotations={this.props.annotations}
            size_x={this.props.width * this.props.width_scale}
            size_y={this.props.height * this.props.height_scale}
            select={this.props.select}
            deselect={this.props.deselect}
            fill_circles={this.props.fill_circles} />
          {hline}
          {vline}
        </svg>
      </div>)
  }
}

export const AnnotationViewX = connector_X(AnnotationView)
export const AnnotationViewY = connector_Y(AnnotationView)
export const AnnotationViewZ = connector_Z(AnnotationView)
