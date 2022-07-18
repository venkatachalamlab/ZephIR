export type worldline_id_t = number
export type worldline_category_t = string

export type color_t = string

export type Worldline_t = {
  id: worldline_id_t,
  name: string,
  // url: string | null,
  // lineage: string | null,
  // description: string | null,
  color: color_t,
  visible: boolean,
}

export type Worldlines_t = {[key: number]: Worldline_t}

export type WorldlinesState_t = {
    worldlines: Worldlines_t,
    worldline_lists: {[key: string]: [worldline_id_t]} | null,
    selected_lists: worldline_category_t[]
}

// This is the model of our module state (e.g. return type of the reducer)
export type State_t = WorldlinesState_t
