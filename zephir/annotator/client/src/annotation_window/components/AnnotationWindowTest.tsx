/* This file should provide an interface to visualize the state and dispatch
actions. UI and styling should be minimal. */

import React from 'react'
import { connect } from "react-redux"
//import { TextField, Button}  from '@material-ui/core';

import * as selectors from '../selectors'
import { actions } from '../actions'
import { saga_actions } from '../sagas'
import { State_t } from '../../app/model'
import { FloatForm } from '../../test_components/ActionForm'

const _Test = ({ state, x_idx, ...A }: any) => {

  const derived_state = { x_idx }

  const full_state = { bare_state: state, derived_state }

  const state_json = JSON.stringify(full_state, null, 4);

  return (
    <div>
      <pre><code>annotation_window: {state_json}</code></pre>

      <br />

      Simple Actions:
      <FloatForm name="set_shape_x" action={A.set_shape_x} />
      <FloatForm name="set_x" action={A.set_x} />
      <FloatForm name="adjust_x" action={A.adjust_x} />
      <FloatForm name="set_scale_x" action={A.set_scale_x} />

      <br />

      Async Actions:
      <br />
      <button onClick={() => A.fetch_state()}>fetch_state</button>

    </div>
  )
}

const mapStateToProps = (state: State_t) => {
  return {
    state: selectors.get_state(state),
    x_idx: selectors.get_x_idx(state)
  }
}

const AnnotationWindowTest =
  connect(mapStateToProps, { ...actions, ...saga_actions })(_Test)

export default AnnotationWindowTest