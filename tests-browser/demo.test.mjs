import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {arrivalsFrom,predict,simulate} from '../demo/core.mjs';

test('browser scores agree with native Python model across deterministic vectors',()=>{
  const vectors=[[0.2,1.5,-0.3],[0],[1e308],[-1e308],...Array.from({length:100},(_,i)=>Array.from({length:i%12+1},(_,j)=>Math.sin(i*7+j)*20))];
  const run=spawnSync(process.env.PYTHON||'python3',['-c',"import json,sys;sys.path.insert(0,'src');from inference_service.model import ToyModel;print(json.dumps(ToyModel(0).predict_batch(json.load(sys.stdin))))"],{input:JSON.stringify(vectors),encoding:'utf8',timeout:20000});
  assert.equal(run.status,0,run.stderr);const native=JSON.parse(run.stdout);
  vectors.forEach((v,i)=>{const browser=predict(v);assert.equal(browser.label,native[i].label);assert.ok(Math.abs(browser.score-native[i].score)<0.000001);assert.equal(browser.model_version,native[i].model_version);});
});
test('queue timeline conserves every request through overload and drains completely',()=>{
  for(let n=1;n<=16;n++){const frames=simulate({arrivals:Array(80).fill(0),maxBatchSize:n,maxQueueSize:4,windowMs:5,latencyMs:20});for(const f of frames){assert.equal(f.received,f.queued+f.batch+f.completed+f.rejected);assert.ok(f.queued<=4);assert.ok(f.batch<=n);}const end=frames.at(-1);assert.equal(end.received,80);assert.equal(end.completed+end.rejected,80);assert.ok(end.rejected>0);}
});
test('zero gathering window uses single-item batches and idle time is preserved',()=>{
  const frames=simulate({arrivals:[0,0,100],maxBatchSize:4,maxQueueSize:4,windowMs:0,latencyMs:10});assert.equal(frames.at(-1).batches,3);assert.equal(frames.at(-1).at,110);assert.equal(frames.at(-1).completed,3);
});
test('demo inputs have enforced bounds and no coercion of feature types',()=>{
  assert.deepEqual(arrivalsFrom('10,0,5'),[0,5,10]);for(const v of ['', '-1','1e2','1001',Array(81).fill(1).join(',')])assert.throws(()=>arrivalsFrom(v));
  for(const v of [[],[true],[null],[Infinity],[1e308,1e308,-1e308]])assert.throws(()=>predict(v));assert.throws(()=>simulate({arrivals:[0],maxBatchSize:0,maxQueueSize:4,windowMs:1,latencyMs:1}));
});
