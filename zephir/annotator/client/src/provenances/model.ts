export type provenance_id_t = string

export type Provenance_t = {
  id: provenance_id_t,
  visible: boolean,
}

export type ProvenancesState_t = {[key: string]: Provenance_t}

// This is the model of our module state (e.g. return type of the reducer)
export type State_t = ProvenancesState_t
