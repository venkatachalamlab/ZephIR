import type { AnnotationWindowState_t } from '../annotation_window/model'
import type { AnnotationsState_t } from '../annotations/model'
import type { WorldlinesState_t } from '../worldlines/model'
import type { ProvenancesState_t } from '../provenances/model'

export type State_t = {
  annotation_window: AnnotationWindowState_t,
  annotations: AnnotationsState_t,
  provenances: ProvenancesState_t,
  worldlines: WorldlinesState_t,
  dataset: string,
};