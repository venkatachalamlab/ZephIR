import React from 'react'
import { bindActionCreators } from 'redux'
import { connect, ConnectedProps } from "react-redux"

import { Card, CardContent } from '@material-ui/core'
import IconButton from '@material-ui/core/IconButton'
import AddIcon from '@material-ui/icons/Add'
import Button from '@material-ui/core/Button'

import { State_t } from '../../app/model'
import { AppDispatch_t } from '../../index'

import { Worldline_t } from '../model'
import * as selectors from '../selectors'
import { actions } from '../actions'
import { saga_actions } from '../sagas'

import { saga_actions as annotation_window_saga_actions }
  from '../../annotation_window'

import Worldline from './Worldline'

const mapStateToProps = (state: State_t) => {
  return {
    state: selectors.get_state(state),
  }
}

const mapDispatchToProps = (dispatch: AppDispatch_t) => {
  return {
    actions: bindActionCreators({ ...actions }, dispatch),
    saga_actions: bindActionCreators({ ...saga_actions }, dispatch),
    set_selected_worldline: bindActionCreators(
      annotation_window_saga_actions.set_selected_worldline, dispatch
    )
  }
}

const connector = connect(mapStateToProps, mapDispatchToProps)
type ReduxProps_t = ConnectedProps<typeof connector>

type WorldlinesPanel_props_t = ReduxProps_t

const _WorldlinesPanel = (props: WorldlinesPanel_props_t) => {

  const worldlines = (Object.values(props.state.worldlines)).map(
    (x: Worldline_t) =>
      <Worldline key={x.id}
        worldline={x}
        on_check={(checked) => {
          props.actions.update_worldline_local({
            id: x.id,
            visible: checked
          })
          if (checked)
            props.set_selected_worldline(x.id)
        }}
        on_rename={(new_name) =>
          props.saga_actions.update_worldline({
            id: x.id,
            name: new_name
          })}
        on_recolor={(new_color) =>
          props.saga_actions.update_worldline({
            id: x.id,
            color: new_color
          })}
      />)

  return (
    <Card
      elevation={3}
      variant="outlined"
      square
      style={{
        position: "relative",
        // left: props.left,
        top: "90px",
        width: "340px",
        height: "auto"
      }}>
      <CardContent>
        <h3>Tracks:</h3>

        <Button
          color="default"
          onClick={(event) => props.saga_actions.set_all_visible(true)}
          style={{
            position: "absolute",
            left: "90px",
            top: "28px"
          }}>
          all
        </Button>
        <Button
          color="default"
          onClick={(event) => props.saga_actions.set_all_visible(false)}
          style={{
            position: "absolute",
            left: "160px",
            top: "28px"
          }}>
          none
        </Button>

        {worldlines}

        <IconButton
          aria-label="add"
          onClick={() => props.saga_actions.create_worldline()}
          style={{
            position: "absolute",
            left: "220px",
            top: "25px"
          }}>
          <AddIcon fontSize="small" />
        </IconButton>

      </CardContent>
    </Card>)

}

const WorldlinesPanel = connector(_WorldlinesPanel)
export default WorldlinesPanel