import React from 'react'
import { bindActionCreators } from 'redux'
import { createNextState } from '@reduxjs/toolkit'
import { connect, ConnectedProps } from "react-redux"
import { get } from 'lodash'

import TextField from '@material-ui/core/TextField'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import Tooltip from '@material-ui/core/Tooltip'
import { withStyles } from '@material-ui/core/styles'
import IconButton from '@material-ui/core/IconButton'
import SaveIcon from '@material-ui/icons/Save'
import GetApp from '@material-ui/icons/GetApp'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import FormControl from '@material-ui/core/FormControl'
import Select from '@material-ui/core/Select'

import { State_t } from '../../app/model'
import { AppDispatch_t } from '../../index'
import { AnnotationWindowState_t } from '../model'
import { actions, saga_actions, selectors } from '../'

const DocstringTooltip = withStyles({
  tooltip: {
    maxWidth: 600,
    margin: 10,
    padding: 10,
    position: "absolute",
    top: -50
  }
})(Tooltip)

const mapStateToProps = (state: State_t) => {
  return {
    window_state: selectors.get_state(state),
  }
}

const mapDispatchToProps = (dispatch: AppDispatch_t) => {
  return {
    actions: bindActionCreators({ ...actions }, dispatch),
    saga_actions: bindActionCreators({ ...saga_actions }, dispatch),
  }
}

const connector = connect(mapStateToProps, mapDispatchToProps)
type ReduxProps_t = ConnectedProps<typeof connector>

type WindowPanel_props_t = ReduxProps_t & {
  left: number,
  top: number,
}

const _WindowPanel = (props: WindowPanel_props_t) => {

  const [window_state, set_window_state] = React.useState(props.window_state)

  React.useEffect(() => {
    set_window_state({
      ...props.window_state,
      x: parseFloat(props.window_state.x.toFixed(3)),
      y: parseFloat(props.window_state.y.toFixed(3)),
      z: parseFloat(props.window_state.z.toFixed(3)),
    });
  },
    [props.window_state])

  const handle_change = (key: string, value: string) => {
    const new_state = { ...(window_state), [key]: value }
    set_window_state(new_state)
  }

  const handle_new_rpc_keybinding = (key: string, rpc_idx: number) => {

    props.actions.set_rpc_keybinding([key, rpc_idx])

    props.actions.set_rpc_arg([key,
      window_state.rpc_list[rpc_idx].default_args])
  }

  const handle_rpc_arg_change = (key: string, value: string) => {

    const rpc_idx = window_state.rpc_keybindings[key]

    const nextState = createNextState(window_state, draft => {
      draft.rpc_args[key] = value
      draft.rpc_list[rpc_idx].default_args = value
    })

    set_window_state(nextState)
  }


  const update = () => {
    props.actions.set_x(window_state.x)
    props.actions.set_y(window_state.y)
    props.actions.set_z(window_state.z)
    props.actions.set_t_idx(window_state.t_idx)

    const a = window_state.selected_annotation
    if ((a === null) || String(a) === "") {
      props.saga_actions.set_selected_annotation(null)
    } else {
      props.saga_actions.set_selected_annotation(Number(a))
    }

    const w = window_state.selected_worldline
    if ((w === null) || String(w) === "") {
      props.saga_actions.set_selected_worldline(null)
    } else {
      props.saga_actions.set_selected_worldline(Number(w))
    }

    props.actions.set_active_provenance(window_state.active_provenance)

    props.actions.set_rpc_args(window_state.rpc_args)
    props.actions.set_rpc_list(window_state.rpc_list)

  }

  const TF = (field: string, name: string) => {
    const field_key = field as keyof AnnotationWindowState_t

    return <TextField
      type="string"
      label={name}
      key={field_key}
      value={window_state[field_key] !== null ? window_state[field_key] : ""}
      onChange={(event) => handle_change(field_key, event.target.value)}
      onBlur={update}
      style={{
        position: "relative",
        width: "80px",
        margin: "10px",
      }}
      InputLabelProps= {{style: { fontSize: 14 }}}
      InputProps={{ style: { fontFamily: "monospace" } }}
    />
  }

  const textfields = [
    ["x", "x"],
    ["y", "y"],
    ["z", "z"],
    ["t_idx", "t"],
    ["selected_annotation", "anno."],
    ["selected_worldline", "track"],
    ["active_provenance", "prov"]]
    .map((x) => TF(x[0], x[1]))

  const RPCKeyConfig = (keyname: string) => {

    const rpc_idx = get(window_state.rpc_keybindings, keyname, 0)
    const rpc = get(window_state.rpc_list, rpc_idx)
    let docstring
    if (rpc) {
      docstring = get(rpc, "docstring", "")
    } else {
      docstring = ""
    }

    return (
      <React.Fragment key={keyname}>
        <DocstringTooltip
          interactive
          title={
            <pre style={{ fontSize: 14 }} >
              {docstring}
            </pre>}
          placement="right"
        >
          <FormControl style={{
            width: 810, margin: 10, top: 10, marginBottom: 50
          }}>
            <InputLabel id={`action-${keyname}-select-label`}>Key {keyname}
            </InputLabel>
            <Select
              labelId={`action-${keyname}-select-label`}
              id={`action-${keyname}-select`}
              value={rpc_idx}
              onChange={(event) => handle_new_rpc_keybinding(
                keyname, Number(event.target.value))}
              style={{
                width: 250,
                position: "absolute",
                fontFamily: "monospace"
              }}
            >
              {props.window_state.rpc_list.map((x, idx) =>
                <MenuItem
                  style={{ fontFamily: "monospace" }}
                  value={idx}
                  key={idx}>
                  {x.name}
                </MenuItem>
              )}
            </Select>
            <TextField
              type="string"
              label={"arg"}
              value={window_state.rpc_args[keyname]}
              onChange={(event) => handle_rpc_arg_change(
                keyname, event.target.value)}
              onBlur={update}
              style={{
                position: "absolute",
                left: 300,
                width: 400,
              }}
              InputProps={{ style: { fontFamily: "monospace" } }}
            />
          </FormControl>
        </DocstringTooltip>
      </React.Fragment>
    )
  }

  const rpc_keys = "1234567890".split("").map(RPCKeyConfig)

  return (
    <Card
      elevation={3}
      variant="outlined"
      square
      style={{
        position: "absolute",
        left: props.left,
        top: props.top,
        width: "840px",
        height: "1000px"
      }}>
      <CardContent>
        <h3>Annotation Window</h3>

        <Tooltip title="Load from disk">
          <IconButton
            aria-label="load"
            onClick={() => props.saga_actions.load()}
            style={{
              position: "absolute",
              left: "650px",
              top: "5px"
            }}>
            <GetApp fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip title="Save to disk">
          <IconButton
            aria-label="save"
            onClick={() => props.saga_actions.save()}
            style={{
              position: "absolute",
              left: "700px",
              top: "5px"
            }}>
            <SaveIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {textfields}

        <Tooltip title="Keyboard shortcut: o">
          <div style={{width:150, float: "left"}}>
            <input type="checkbox"
              checked={props.window_state.fill_circles}
              id="fill-circles-cb"
              onChange={(event) =>
                props.actions.set_fill_circles(event.target.checked)}
              style={{
                // float: "left",
                marginTop: "30px"
              }} />
            <label>Fill circles</label>
          </div>
        </Tooltip>

        <Tooltip title="Keyboard shortcut: a">
          <div style={{width:150, float: "left"}}>
            <input type="checkbox"
              checked={props.window_state.show_all_annotations}
              id="all-annotations-cb"
              onChange={(event) =>
                props.actions.set_show_all_annotations(event.target.checked)}
              style={{
                // float: "left",
                marginTop: "30px"
              }} />
            <label>Show all</label>
          </div>
        </Tooltip>


        {rpc_keys}

      </CardContent>
    </Card>)
}

const WindowPanel = connector(_WindowPanel)
export default WindowPanel