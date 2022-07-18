import { createSlice, PayloadAction } from '@reduxjs/toolkit'

import {
  AnnotationsState_t as State_t ,
  Annotation_t,
  annotation_id_t
} from './model'

export const initialState: State_t = {}

const reducers = {

  set_state: (state: State_t, action: PayloadAction<State_t>) =>
    action.payload,

  set_annotation_local: (
    state: State_t,
    action: PayloadAction<Annotation_t>) => {
      state[action.payload.id] = action.payload
    },

  delete_annotation_local: (
    state: State_t,
    action: PayloadAction<annotation_id_t>) => {
      delete state[action.payload]
    }

}

const annotations_slice = createSlice({
  name: 'annotations',
  initialState,
  reducers
})

export const { actions, reducer } = annotations_slice