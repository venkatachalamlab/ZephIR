/* This file should provide an interface to visualize the state and dispatch
actions. UI and styling should be minimal. */

import React from 'react'
import { connect } from "react-redux"
import ReactJson from 'react-json-view'

import { actions } from '../actions'
import { saga_actions } from '../sagas'
import { State_t } from '../../app/model'
import { FloatForm } from '../../test_components/ActionForm'

import * as annotation_window_selectors
  from '../../annotation_window/selectors'

const _Test = ({ state, x_idx, ...A }: any) => {

  const derived_state = { x_idx }

  const full_state = { bare_state: state, derived_state }

  // const state_json = JSON.stringify(full_state, null, 4);

  return (
    <div>

      {/* <pre><code>app: {state_json}</code></pre> */}
      <ReactJson theme="monokai" src={full_state} />

      <br />

      Simple Actions:
      <FloatForm name="set_shape_x" action={A.set_shape_x} />
      <FloatForm name="set_x" action={A.set_x} />
      <FloatForm name="set_t_idx" action={A.set_t_idx} />
      <FloatForm name="adjust_x" action={A.adjust_x} />
      <FloatForm name="set_scale_x" action={A.set_scale_x} />

      <br />

      Async Actions:
      <br />

      <button onClick={() => A.fetch_state()}>
        fetch_state
      </button>

      <button onClick={() => A.insert_annotation_here()}>
        insert_annotation
      </button>

      <FloatForm name="fetch_annotations" action={A.get_annotations} />

    </div>
  )
}

const mapStateToProps = (state: State_t) => {
  return {
    state: state,
    x_idx: annotation_window_selectors.get_x_idx(state)
  }
}

const AppTest =
  connect(mapStateToProps,
    { ...actions.annotation_window,
      ...actions.annotations,
      ...saga_actions.annotation_window,
      ...saga_actions.annotations
    })(_Test)

export default AppTest