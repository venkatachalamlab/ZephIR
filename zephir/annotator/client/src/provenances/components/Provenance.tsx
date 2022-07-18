import React from 'react'

import { Provenance_t } from '../model'

type Provenance_props_t = {
  provenance: Provenance_t,
  on_check: (state: boolean) => any,
}

const Provenance = (props: Provenance_props_t) => {

  return (
    <div
      style={{
        float: "left",
        position: "relative",
        height: "30px",
        width: "100px"
      }}>
      <input type="checkbox"
        checked={props.provenance.visible}
        onChange={(event) => props.on_check(event.target.checked)}
        style={{
          float: "left",
          marginTop: "8px"
        }} />
      <div
        style={{
          float: "left",
          fontFamily: "monospace",
          fontSize: "14px",
          color: "#C0C0C0",
          marginTop: "5px",
          border: "0px",
          width: "20px"
        }}>
          { props.provenance.id }
      </div>
    </div >)

}

export default Provenance