export function predict(values){if(!Array.isArray(values)||values.length<1||values.length>256||values.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error('Use 1–256 finite numeric features.');const logit=values.reduce((sum,v,i)=>sum+(i+1)*v,0)/values.length;if(Number.isNaN(logit))throw Error('Weighted sum is undefined after overflow; use smaller feature values.');const score=logit>=0?1/(1+Math.exp(-logit)):Math.exp(logit)/(1+Math.exp(logit));return {label:score>=0.5?'positive':'negative',score:Number(score.toFixed(6)),model_version:'toy-logistic-v1'};}
export function arrivalsFrom(source){if(source.length>500)throw Error('Arrival list exceeds 500 characters.');const parts=source.split(',').map(s=>s.trim());if(parts.length>80||parts.some(s=>!/^\d+$/.test(s)||Number(s)>1000))throw Error('Use 1–80 comma-separated whole arrival times from 0 to 1,000 ms.');return parts.map(Number).sort((a,b)=>a-b);}
export function simulate({arrivals,maxBatchSize,maxQueueSize,windowMs,latencyMs}){
  for(const [name,v,min,max] of [['batch size',maxBatchSize,1,16],['queue slots',maxQueueSize,1,64],['window',windowMs,0,100],['model time',latencyMs,1,200]])if(!Number.isInteger(v)||v<min||v>max)throw Error(`${name} must be a whole number from ${min} to ${max}.`);
  if(!Array.isArray(arrivals)||arrivals.length<1||arrivals.length>80||arrivals.some(v=>!Number.isInteger(v)||v<0||v>1000))throw Error('Invalid bounded arrival times.');
  const incoming=arrivals.map((at,i)=>({at,id:i+1})).sort((a,b)=>a.at-b.at||a.id-b.id),frames=[];let queue=[],batch=[],phase='idle',due=Infinity,at=0,index=0,completed=0,rejected=0,received=0,batches=0;
  const emit=event=>frames.push({at,event,queued:queue.length,batch:batch.length,completed,rejected,received,batches});
  function dispatch(){phase='running';due=at+latencyMs;batches++;emit(`Start batch: requests ${batch.map(x=>x.id).join(', ')}`);}
  function gather(){if(phase==='idle'&&queue.length){phase='gathering';batch.push(queue.shift());due=at+windowMs;}if(phase==='gathering'){// With a zero window the Python worker does not poll further queue items.
      if(windowMs>0)while(queue.length&&batch.length<maxBatchSize)batch.push(queue.shift());
      if(batch.length>=maxBatchSize||due<=at)dispatch();}}
  let guard=0;
  while(index<incoming.length||queue.length||batch.length){if(++guard>1000)throw Error('Timeline work limit exceeded.');at=Math.min(incoming[index]?.at??Infinity,due);
    if(due<=at){if(phase==='running'){completed+=batch.length;batch=[];phase='idle';due=Infinity;emit('Batch completed');gather();}else if(phase==='gathering')dispatch();}
    while(index<incoming.length&&incoming[index].at===at){const request=incoming[index++];received++;if(queue.length>=maxQueueSize){rejected++;emit(`Reject request ${request.id}: waiting queue full`);}else{queue.push(request);gather();emit(`Admit request ${request.id}`);}}
    gather();
  }
  return frames;
}
