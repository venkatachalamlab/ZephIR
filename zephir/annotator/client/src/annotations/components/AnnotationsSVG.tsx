import React from 'react'

import { AnnotationView2D_t } from '../selectors'
import { Dictionary } from 'lodash'

type AnnotationSVG_props_t = AnnotationView2D_t & {
  size_x: number,
  size_y: number,
  select: (id: number) => any,
  deselect: (id: number) => any,
  fill_circles: boolean,
}

const AnnotationSVG = (props: AnnotationSVG_props_t) =>
  <circle
    key={props.id}
    cx={props.x * props.size_x}
    cy={props.y * props.size_y}
    r={props.selected ? 2 * props.radius : props.radius}
    stroke={props.fill_circles? "white":  props.color}
    strokeWidth="1"
    fill={props.fill_circles? props.color: "none"}
    onClick={(e) => {
      props.selected ? props.deselect(props.id) : props.select(props.id)
      e.preventDefault()
      e.stopPropagation()
    }}
    onDoubleClick={(e) => { e.preventDefault(); e.stopPropagation(); }} />


type AnnotationsSVG_props_t = {
  annotations: Dictionary<AnnotationView2D_t>
  size_x: number,
  size_y: number,
  select: (id: number) => any,
  deselect: (id: number) => any,
  fill_circles: boolean,
}

const AnnotationsSVG = (props: AnnotationsSVG_props_t) => {

  const annotation_SVGs = Object.entries(props.annotations).map((a) =>
    AnnotationSVG({
      ...a[1],
      size_x: props.size_x,
      size_y: props.size_y,
      select: props.select,
      deselect: props.deselect,
      fill_circles: props.fill_circles
    }))

  return <g> {annotation_SVGs} </g>
}

export default AnnotationsSVG