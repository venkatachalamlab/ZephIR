import React, { useState } from 'react'

import { Worldline_t } from '../model'

type Worldline_props_t = {
  // left: number,
  // top: number,
  worldline: Worldline_t,
  on_check: (state: boolean) => any,
  on_rename: (new_name: string) => any,
  on_recolor: (new_color: string) => any,
}

const Worldline = (props: Worldline_props_t) => {

  const get_name = (x: Worldline_props_t) => {
    const name = x.worldline.name
    if (name && name !== "null") {
      return name
    } else {
      return String(x.worldline.id)
    }
  }

  const [state, setState] = useState(get_name(props))
  React.useEffect(() => {
    setState(get_name(props));
  }, [props])


  return (
    <div
      style={{
        float: "left",
        position: "relative",
        height: "30px",
        width: "100px"
        // left: props.left,
        // top: props.top,
      }}>
      <input type="checkbox"
        checked={props.worldline.visible}
        onChange={(event) => props.on_check(event.target.checked)}
        style={{
          float: "left",
          marginTop: "8px"
        }} />
      <div
        style={{
          float: "left",
          fontFamily: "monospace",
          fontSize: "10px",
          color: "#C0C0C0",
          marginTop: "8px",
          border: "0px",
          width: "20px"
        }}>
          { props.worldline.id }
      </div>
      <input type="text"
        onBlur={(event) => props.on_rename(event.target.value)}
        onChange={(event) => setState(event.target.value)}
        value={state}
        style={{
          float: "left",
          marginTop: "5px",
          fontFamily: "monospace",
          color: "white",
          backgroundColor: "#666",
          border: "0px",
          width: "40px"
        }} />
      <div style={{
        backgroundColor: props.worldline.color,
        float: "left",
        marginTop: "5px",
        border: "0px",
        padding: "0px",
        width: "16px",
        height: "18px",
      }} >
        <input type="color"
          value={props.worldline.color}
          onChange={(event) => props.on_recolor(event.target.value)}
          style={{
            float: "left",
            opacity: "0",
            marginTop: "5px",
            border: "0px",
            padding: "0px",
            width: "16px",
            height: "16px",
          }} />
      </div>
    </div >)

}

export default Worldline