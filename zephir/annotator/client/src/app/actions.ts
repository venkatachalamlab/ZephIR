import { combineReducers } from 'redux'

import {
  actions as annotation_window_actions,
  reducer as annotation_window_reducer
} from '../annotation_window'

import {
  actions as annotations_actions,
  reducer as annotations_reducer
} from '../annotations'

import {
  actions as worldlines_actions,
  reducer as worldlines_reducer
} from '../worldlines'

import {
  actions as provenances_actions,
  reducer as provenances_reducer
} from '../provenances'

export const actions = {
  annotation_window: annotation_window_actions,
  annotations: annotations_actions,
  worldlines: worldlines_actions,
  provenances: provenances_actions
}

export const reducer = combineReducers({
  annotation_window: annotation_window_reducer,
  annotations: annotations_reducer,
  worldlines: worldlines_reducer,
  provenances: provenances_reducer
});