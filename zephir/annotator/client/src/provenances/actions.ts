import { createSlice, PayloadAction } from '@reduxjs/toolkit'

import {
  provenance_id_t,
  ProvenancesState_t as State_t,
} from './model'

export const initialState: State_t = {}

const reducers = {

  set_state: (_state: State_t, action: PayloadAction<State_t>) =>
    action.payload,

  set_provenance_selected:  (
    state: State_t,
    action: PayloadAction<[provenance_id_t, boolean]>) => {
    state[action.payload[0]].visible = action.payload[1]
  },

  set_all_selected: (
    state: State_t,
    action: PayloadAction<boolean>) => {
      for (let provenance in state) {
        state[provenance].visible = action.payload
      }
    }

}

const provenances_slice = createSlice({
  name: 'provenances',
  initialState,
  reducers
})

export const { actions, reducer } = provenances_slice