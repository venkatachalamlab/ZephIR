import React from 'react'
import { bindActionCreators } from 'redux'
import { connect, ConnectedProps } from "react-redux"

import { Slider, TextField } from '@material-ui/core';

import { AppDispatch_t } from '../../index'
import { actions } from '../actions'
import * as selectors from '../selectors'
import { saga_actions as annotations_saga_actions } from '../../annotations'
import { selectors as annotations_selectors } from '../../annotations'
import { State_t } from '../../app/model'
import { AnnotationViewZ, AnnotationViewX, AnnotationViewY }
  from './AnnotationView'
import { AnnotationPanel } from '../../annotations'
import { saga_actions } from '../sagas'
import WindowPanel from './WindowPanel'
import { WorldlinesPanel } from '../../worldlines'
import { VolumeView } from './VolumeView'
import { ProvenancePanel } from '../../provenances'

const XYZSliderBase = (props: {
  left: number,
  top: number,
  length: number,
  flip: boolean,
  orientation: "horizontal" | "vertical",
  val: number,
  handler: (x: number) => any
}) => {

  return <Slider
    style={{
      position: "absolute",
      left: props.left,
      top: props.top,
      width: (props.orientation === "horizontal") ? props.length : 0,
      height: (props.orientation === "vertical") ? props.length : 0
    }}
    color="primary"
    orientation={props.orientation}
    defaultValue={0.5}
    value={props.flip ? (1 - props.val) : props.val}
    min={0}
    max={1}
    step={0.0001}
    track={false}
    scale={props.flip ? (x) => (1 - x) : (x) => x}
    onChange={(e, v
    ) => props.handler(props.flip ? 1 - (v as number) : (v as number))} />
}

const TSliderBase = (props: {
  left: number,
  top: number,
  length: number,
  val: number,
  max_val: number,
  handler: (x: number) => any
}) => {

  return <Slider
    style={{
      position: "absolute",
      left: props.left,
      top: props.top,
      width: props.length,
    }}
    color="secondary"
    orientation="horizontal"
    defaultValue={0}
    value={props.val}
    min={0}
    max={props.max_val}
    track={false}
    step={1}
    onChange={(e, v) => props.handler(v as number)} />
}

const LUTSliderBase = (props: {
  left: number,
  top: number,
  length: number,
  color: string,
  val: [number, number],
  handler: (x: [number, number]) => any
}) => {
  const [value, setValue] = React.useState<number[]>([0, 255]);

  const handleChange = (event: any, newValue: number | number[]) => {
    setValue(newValue as number[]);
    props.handler(newValue as [number, number])
  };

  return (

    <Slider
      style={{
        position: "absolute",
        left: props.left,
        top: props.top,
        width: props.length,
        color: props.color,
      }}
      value={value}
      min={0}
      max={255}
      onChange={handleChange}
      valueLabelDisplay="auto"
    />

  );
}

const GammaBoxBase = (props: {
  left: number,
  top: number,
  length: number,
  val: number,
  handler: (x: number) => any
}) => {

  return (

    <TextField
      variant={"outlined"}
      style={{
        position: "absolute",
        left: props.left,
        top: props.top,
        width: props.length,
      }}
      label={"\u03B3"}
      defaultValue={1.0}
      value={props.val}
      type={"number"}
      inputProps={{
        maxLength: 3,
        step: "0.1"
      }}
      onChange={(e
      ) => props.handler(parseFloat(e.target.value).toFixed(1) as unknown as number)}
    />

  );
}

const mapStateToProps = (state: State_t) => {
  return {
    window_state: selectors.get_state(state),
    annotations_z: annotations_selectors.get_z_annotations(state),
    annotations_3D: annotations_selectors.get_annotations_view3D(state),
    size_x: selectors.get_shape_x(state) * selectors.get_scale_x(state),
    size_y: selectors.get_shape_y(state) * selectors.get_scale_y(state),
    size_z: selectors.get_shape_z(state) * selectors.get_scale_z(state),
    x: selectors.get_x(state),
    y: selectors.get_y(state),
    z: selectors.get_z(state),
    t_idx: selectors.get_t_idx(state),
    shape_t: selectors.get_shape_t(state),
    LUT: selectors.get_LUT(state),
    gamma: selectors.get_gamma(state),
    view: selectors.get_view(state),
  }
}

const mapDispatchToProps = (dispatch: AppDispatch_t) => {
  return {
    actions: bindActionCreators({ ...actions }, dispatch),
    saga_actions: bindActionCreators({ ...saga_actions }, dispatch),
    delete_annotation: () => dispatch(
      annotations_saga_actions.delete_annotation()),
  }
}

const connector = connect(mapStateToProps, mapDispatchToProps)
type ReduxProps_t = ConnectedProps<typeof connector>

type AnnotationWindow_props_t = ReduxProps_t & {
}

const _AnnotationWindow = (props: AnnotationWindow_props_t) => {

  const origin_left = 20;
  const origin_top = 20;
  const padding_1 = 10;

  const x_image_left = origin_left + props.size_x + padding_1;
  const x_image_top = origin_top;

  const y_image_left = origin_left;
  const y_image_top = origin_top + props.size_y + padding_1;

  const x_slider_left = origin_left;
  const x_slider_top = y_image_top + props.size_z;

  const y_slider_left = x_image_left + props.size_z + 2 * padding_1;
  const y_slider_top = origin_top;

  const z_slider_1_left = x_image_left;
  const z_slider_1_top = x_slider_top;

  const z_slider_2_left = y_slider_left;
  const z_slider_2_top = y_slider_top + y_image_top - origin_top;

  const t_slider_left = origin_left
  const t_slider_top = x_slider_top + 2 * padding_1
  const t_slider_length = props.size_x + props.size_z + padding_1

  const annotation_panel_left = y_slider_left + 5 * padding_1

  const LUT_slider_length = t_slider_length / 3 - 6 * padding_1
  const LUT_top = t_slider_top + 5 * padding_1
  const LUT_R_left = origin_left
  const LUT_G_left = LUT_R_left + LUT_slider_length + 4 * padding_1
  const LUT_B_left = LUT_G_left + LUT_slider_length + 4 * padding_1
  const Gamma_left = LUT_B_left + LUT_slider_length + 4 * padding_1

  const window_panel_left = origin_left
  const window_panel_top = LUT_top + 5 * padding_1


  let views
  if (props.view === "vol") {
    views = (
      <VolumeView
        left={origin_left}
        top={origin_top} />
    )
  }
  else {
    views = (
      <React.Fragment>
        <AnnotationViewZ left={origin_left} top={origin_top} />
        <AnnotationViewX left={x_image_left} top={x_image_top} />
        <AnnotationViewY left={y_image_left} top={y_image_top} />
      </React.Fragment>
    )
  }

  const XSlider =
    <XYZSliderBase
      left={x_slider_left}
      top={x_slider_top}
      length={props.size_x}
      orientation="horizontal"
      flip={false}
      val={props.x}
      handler={props.actions.set_x} />

  const YSlider =
    <XYZSliderBase
      left={y_slider_left}
      top={y_slider_top}
      length={props.size_y}
      orientation="vertical"
      flip={true}
      val={props.y}
      handler={props.actions.set_y} />

  const ZSlider1 =
    <XYZSliderBase
      left={z_slider_1_left}
      top={z_slider_1_top}
      length={props.size_z}
      orientation="horizontal"
      flip={false}
      val={props.z}
      handler={props.actions.set_z} />

  const ZSlider2 =
    <XYZSliderBase
      left={z_slider_2_left}
      top={z_slider_2_top}
      length={props.size_z}
      orientation="vertical"
      flip={true}
      val={props.z}
      handler={props.actions.set_z} />

  const TSlider =
    <TSliderBase
      left={t_slider_left}
      top={t_slider_top}
      length={t_slider_length}
      val={props.t_idx}
      max_val={props.shape_t - 1}
      handler={props.actions.set_t_idx} />

  const LUTSliderRed =
    <LUTSliderBase
      left={LUT_R_left}
      top={LUT_top}
      length={LUT_slider_length}
      val={props.LUT.red}
      color="red"
      handler={([l, h]) => props.actions.set_LUT_color(["red", l, h])} />
  const LUTSliderGreen =
    <LUTSliderBase
      left={LUT_G_left}
      top={LUT_top}
      length={LUT_slider_length}
      val={props.LUT.green}
      color="green"
      handler={([l, h]) => props.actions.set_LUT_color(["green", l, h])} />
  const LUTSliderBlue =
    <LUTSliderBase
      left={LUT_B_left}
      top={LUT_top}
      length={LUT_slider_length}
      val={props.LUT.blue}
      color="blue"
      handler={([l, h]) => props.actions.set_LUT_color(["blue", l, h])} />

  const gammaBox =
    <GammaBoxBase
      left={Gamma_left}
      top={LUT_top - 2*padding_1}
      length={LUT_slider_length / 3}
      val={props.gamma}
      handler={props.actions.set_gamma} />

  const keydown_handler = (event: React.KeyboardEvent) => {

    const key = String.fromCharCode(event.which)
    const ctrl = event.ctrlKey
    const shift = event.shiftKey

    switch (key) {
      case "F": {
        if (shift) props.saga_actions.adjust_t(10)
        else props.saga_actions.adjust_t(1)
        break;
      }
      case "D": {
        if (shift) props.saga_actions.adjust_t(-10)
        else props.saga_actions.adjust_t(-1)
        break;
      }
      case "V": {
        if (shift) props.actions.set_z(props.z + 0.2)
        else if (ctrl) props.actions.set_z(props.z + 0.02)
        else props.actions.set_z(props.z + 0.05)
        break;
      }
      case "C": {
        if (shift) props.actions.set_z(props.z - 0.2)
        else if (ctrl) props.actions.set_z(props.z - 0.01)
        else props.actions.set_z(props.z - 0.05)
        break;
      }
      case "R": {
        props.actions.adjust_c_idx(1)
        break
      }
      case "E": {
        props.actions.adjust_c_idx(-1)
        break
      }
      case "W": {
        props.saga_actions.adjust_selected_worldline(1)
        break
      }
      case "Q": {
        props.saga_actions.adjust_selected_worldline(-1)
        break
      }
      case ".": {
        props.delete_annotation()
        break
      }
      case "S": {
        if (ctrl) props.saga_actions.save()
        break
      }
      case "O": {
        props.actions.toggle_fill_circles()
        break
      }
      case "A": {
        props.actions.toggle_all_annotations()
        break
      }
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
      case "0":
        if (!ctrl && !shift) props.saga_actions.rpc_from_key(key)
        break
    }

    event.preventDefault()
  }

  const keyup_handler = (event: React.KeyboardEvent) => {
    //const key = String.fromCharCode(event.which)
  }

  return (
    <div style={{ position: "absolute" }}>
      <div tabIndex={0}
        onKeyDown={keydown_handler}
        onKeyUp={keyup_handler}>

        {views}

        {XSlider}
        {YSlider}
        {ZSlider1}
        {ZSlider2}
        {TSlider}

        {LUTSliderRed}
        {LUTSliderGreen}
        {LUTSliderBlue}

        {gammaBox}

      </div>

      <WindowPanel
        left={window_panel_left}
        top={window_panel_top} />

      <div style={{ position: "absolute",
        left: annotation_panel_left,
        width: "340px" }}>

        <AnnotationPanel/>

        <ProvenancePanel/>

        <WorldlinesPanel/>

      </div>


    </div>
  )
}

const AnnotationWindow = connector(_AnnotationWindow)

export default AnnotationWindow