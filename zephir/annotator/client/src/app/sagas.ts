import { all } from 'redux-saga/effects'

import {
  sagas as annotation_window_sagas,
  saga_actions as annotation_window_saga_actions,
} from '../annotation_window'

import {
  sagas as annotations_sagas,
  saga_actions as annotations_saga_actions,
} from '../annotations'

import {
  sagas as worldlines_sagas,
  saga_actions as worldlines_saga_actions,
} from '../worldlines'

import {
  sagas as provenances_sagas,
  saga_actions as provenances_saga_actions,
} from '../provenances'

export function* saga() {
  yield all([
    ...annotation_window_sagas,
    ...annotations_sagas,
    ...worldlines_sagas,
    ...provenances_sagas
  ].map(f => f()))
}

export const saga_actions = {
  annotation_window: annotation_window_saga_actions,
  annotations: annotations_saga_actions,
  worldlines: worldlines_saga_actions,
  provenances: provenances_saga_actions
}