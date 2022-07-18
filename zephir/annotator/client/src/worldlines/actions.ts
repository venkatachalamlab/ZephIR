import { createSlice, PayloadAction } from '@reduxjs/toolkit'

import {
  worldline_id_t,
  Worldline_t,
  Worldlines_t,
  WorldlinesState_t as State_t,
} from './model'

export const initialState: State_t = {
  worldlines: {},
  worldline_lists: null,
  selected_lists: []
}

const reducers = {

  set_state: (state: State_t, action: PayloadAction<State_t>) =>
    action.payload,

  set_worldlines_local: (
    state: State_t,
    action: PayloadAction<Worldlines_t>) => {
    state.worldlines = action.payload
  },

  set_worldline_lists_local: (
    state: State_t,
    action: PayloadAction<{ [key: string]: [worldline_id_t] }>) => {
    state.worldline_lists = action.payload
    state.selected_lists = []
  },

  set_worldline_local: (
    state: State_t,
    action: PayloadAction<Worldline_t>) => {
    state.worldlines[action.payload.id] = action.payload
  },

  update_worldline_local: (
    state: State_t,
    action: PayloadAction<Partial<Worldline_t>>) => {
    if (action.payload.id !== undefined)
      state.worldlines[action.payload.id] = {
        ...state.worldlines[action.payload.id],
        ...action.payload
      }
  },

  delete_worldline_local: (
    state: State_t,
    action: PayloadAction<worldline_id_t>) => {
    delete state.worldlines[action.payload]
  },

}

const worldlines_slice = createSlice({
  name: 'worldlines',
  initialState,
  reducers
})

export const { actions, reducer } = worldlines_slice