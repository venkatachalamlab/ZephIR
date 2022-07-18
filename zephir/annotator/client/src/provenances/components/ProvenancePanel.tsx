import React from 'react'
import { bindActionCreators } from 'redux'
import { connect, ConnectedProps } from "react-redux"

import { Card, CardContent } from '@material-ui/core'
import IconButton from '@material-ui/core/IconButton'
import RefreshIcon from '@material-ui/icons/Refresh'
import Button from '@material-ui/core/Button'

import { State_t } from '../../app/model'
import { AppDispatch_t } from '../../index'

import { Provenance_t } from '../model'
import * as selectors from '../selectors'
import { actions } from '../actions'
import { saga_actions } from '../sagas'

import { saga_actions as annotation_window_saga_actions }
  from '../../annotation_window'

import Provenance from './Provenance'

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

type ProvenancePanel_props_t = ReduxProps_t

const _ProvenancePanel = (props: ProvenancePanel_props_t) => {

  const provenances = (Object.values(props.state)).map(
    (x: Provenance_t) =>
      <Provenance key={x.id}
        provenance={x}
        on_check={(checked) =>
          props.actions.set_provenance_selected([x.id, checked])
        }
      />)

  return (
    <Card
      elevation={3}
      variant="outlined"
      square
      style={{
        position: "relative",
        top: "50px",
        width: "340px",
        height: "auto"
      }}>
      <CardContent>
        <h3>Provenances:</h3>

        <Button
          color="default"
          onClick={(event) => props.actions.set_all_selected(true)}
          style={{
            position: "absolute",
            left: "140px",
            top: "28px"
          }}>
          all
        </Button>
        <Button
          color="default"
          onClick={(event) => props.actions.set_all_selected(false)}
          style={{
            position: "absolute",
            left: "190px",
            top: "28px"
          }}>
          none
        </Button>

        {provenances ? provenances : null}

        <IconButton
          aria-label="fetch"
          onClick={props.saga_actions.fetch_provenances}
          style={{
            position: "absolute",
            left: "250px",
            top: "25px"
          }}>
          <RefreshIcon fontSize="small" />
        </IconButton>

      </CardContent>
    </Card>)

}

const ProvenancePanel = connector(_ProvenancePanel)
export default ProvenancePanel