import React, {useEffect} from 'react'
import { useDispatch } from "react-redux";

import { AnnotationWindow } from '../../annotation_window'


const App = () => {

  // Hacky listener for RPC-like requests routed through localhost/socket
  // Constantly listens for new message on a WebSocket
  // Dispatches actions as specified in the message
  const dispatch = useDispatch();
  useEffect(() => {
    const loc = 'ws://' + window.location.host + '/listen';
    const es = new EventSource(loc);
    es.onopen = () => {
        console.log('Connecting to socket: ' + loc)
    }
    es.onmessage = (msg) => {
        console.log('Received msg: ' + msg.data);
        const result = JSON.parse(msg.data);
        for (let action of result) {
          if (action.type !== "handshake") {
            const try_int_payload = parseInt(action.payload)
            if (String(try_int_payload) === action.payload) {
              action.payload = try_int_payload
            }
            dispatch(action)
          }
        }
    }
    return () => {
      console.log("closing")
    };
  }, [] );

  return (
    <div>
      <AnnotationWindow />
    </div>
  );
}

export default App