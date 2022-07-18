import React from 'react'
import { bindActionCreators } from 'redux'
import { connect, ConnectedProps } from "react-redux"
import { TextField, Card, CardContent } from '@material-ui/core'
import IconButton from '@material-ui/core/IconButton'
import DeleteIcon from '@material-ui/icons/Delete'
import DoneIcon from '@material-ui/icons/Done'

import { State_t } from '../../app/model'
import { AppDispatch_t } from '../../index'
import * as selectors from '../selectors'
import { saga_actions } from '../sagas'
import { Annotation_t } from '../model'

const mapStateToProps = (state: State_t) => {
  return {
    annotation: selectors.get_selected_annotation(state),
  }
}

const mapDispatchToProps = (dispatch: AppDispatch_t) => {
  return {
    saga_actions: bindActionCreators({ ...saga_actions }, dispatch),
  }
}

const connector = connect(mapStateToProps, mapDispatchToProps)
type ReduxProps_t = ConnectedProps<typeof connector>

type AnnotationPanel_props_t = ReduxProps_t

const _AnnotationPanel = (props: AnnotationPanel_props_t) => {

  const [annotation, setAnnotation] = React.useState(props.annotation)

  React.useEffect(() => {
    setAnnotation(props.annotation);
  },
    [props.annotation])

  let textfields = [<div key="null" style={{ position: "absolute" }}>
    select an annotation
    </div>,]

  const handle_change = (key: string, value: number | string) => {
    const new_annotation = { ...(annotation as Annotation_t), [key]: value }
    console.log(key, value, new_annotation)
    setAnnotation(new_annotation)
  }

  if (annotation != null) {

    const TF = (field: string, name: string) => {
      const field_key = field as keyof Annotation_t

      return <TextField
        type="string"
        label={name}
        key={field_key}
        value={annotation[field_key] !== null ? annotation[field_key] : ""}
        onChange={(event) => handle_change(field_key, event.target.value)}
        style={{
          position: "relative",
          width: "100px",
          height: "30px",
          margin: "10px",
        }}
        InputProps={{ style: { fontFamily: "monospace" } }}
      />
    }

    textfields = [
      ["x", "x"],
      ["y", "y"],
      ["z", "z"],
      ["t_idx", "t"],
      ["id", "id"],
      ["worldline_id", "track"],
      ["parent_id", "parent_id"],
      ["provenance", "provenance"]]
      .map((x) => TF(x[0], x[1]))

  }

  return (
    <Card
      elevation={3}
      variant="outlined"
      square
      style={{
        position: "relative",
        width: "340px",
        height: "300px",
        top: "20px"
      }}>
      <CardContent>
        <h3>Annotation</h3>

        <IconButton
          aria-label="done"
          onClick={() =>
            props.saga_actions.update_annotation(annotation as Annotation_t)}
          style={{
            position: "absolute",
            left: "230px",
            top: "20px"
          }}>
          <DoneIcon fontSize="small" />
        </IconButton>

        <IconButton
          aria-label="delete"
          onClick={props.saga_actions.delete_annotation}
          style={{
            position: "absolute",
            left: "280px",
            top: "20px"
          }}>
          <DeleteIcon fontSize="small" />
        </IconButton>

        {textfields}
      </CardContent>
    </Card>)
}

const AnnotationPanel = connector(_AnnotationPanel)
export default AnnotationPanel