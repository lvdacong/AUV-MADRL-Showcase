/* Scene adapter: reuse the existing cards, controls, inspectors, charts and canvases. */
function passageLabel(config) {
  const left=100*(config?.left_probability??.5);
  const percent=value=>Number(value.toFixed(2));
  return `L:R ${percent(left)}:${percent(100-left)} · Seed ${config?.seed??'?'}`;
}
function createPassageUI({send, bars, stateBars, beforeSwitch, inspection, segments, rates, source}) {
  const el=id=>document.getElementById(id);
  const cmd=(action,extra={})=>send({type:'passage',action,...extra});
  let data=null, mode='continuous', signature='', requestedMode=null, initialized=false;
  const fields={environments:'training-parallel-environments',rollout_steps:'training-rollout',
    gamma:'training-gamma',gae_lambda:'training-gae-lambda',maximum_episode_steps:'training-tmax'};
  const original=new Map();
  const style=document.createElement('style');
  style.textContent=`
    .scene-unavailable{opacity:.4}
    .scene-unavailable .scene-unavailable{opacity:1}
    body[data-scene-mode=discrete] #selected-policy-output{display:block}
    body[data-scene-mode=discrete] #selected-policy-output .actor-compact-headings,body[data-scene-mode=discrete] #selected-policy-output .actor-compact-bars{grid-template-columns:repeat(4,minmax(0,1fr))}
  `;
  document.head.append(style);
  const switcher=document.createElement('label');switcher.className='rate-control';
  switcher.innerHTML='<span>Scene</span><select id="scene-mode" aria-label="Scene mode"><option value="continuous">Continuous</option><option value="discrete">Discrete</option></select>';
  el('toolbar-primary').prepend(switcher);
  el('scene-mode').onchange=event=>{beforeSwitch();send({type:'set_scene_mode',value:event.target.value});};
  const network=document.querySelector('.compact-settings[data-section="Scene / Network"]');
  const add=(parent,label,id,html)=>{
    const row=document.createElement('label');row.className='range-setting passage-only';
    row.innerHTML='<span class="field-label">'+label+'</span>'+html;parent.append(row);return el(id);
  };
  add(document.querySelector('.compact-settings[data-section="Match"]'),'End on collision','passage-collision','<input id="passage-collision" type="checkbox">').onchange=()=>settings();
  const hideIds=['target-mode','first-casualty','defender-elimination','reward-objective','policy-sharing','joint-training-mode','alternating-updates','apf-attraction','apf-repulsion','viewer-sampling-limit','remaining-time-observation','world-map-seed','world-map-observation','entity-streams','entity-self-attention-layers','defender-spawn-radius-fraction','engagement-radius-fraction','attacker-spawn-inner-radius-fraction','obstacle-radius-mean','obstacle-radius-std','obstacle-density-mean','obstacle-density-std'];
  for(const id of hideIds) el(id)?.closest('.range-setting')?.classList.add('continuous-only');
  for(const id of ['world-map-preview','main-view-frame-control','main-view-zoom-control','scene-lock','replay-last','capture-end','performance-status','horizon-status','selected-maps','selected-entities','context-status']) el(id)?.classList.add('continuous-only');
  network.querySelector('.panel-hint')?.classList.add('continuous-only');
  for(const row of document.querySelectorAll('.panel-row'))if([...row.children].length&&[...row.children].every(n=>n.classList.contains('continuous-only')))row.classList.add('continuous-only');
  document.querySelector('#role-setting-rows [data-role=learner] [data-value=rule]').classList.add('continuous-only');
  document.querySelectorAll('#role-setting-rows [data-role]:not([data-role=learner])').forEach(node=>node.classList.add('continuous-only'));
  for(const id of [...Object.values(fields),'training-updates','training-entropy'])el(id).addEventListener('change',event=>{if(active()){event.stopImmediatePropagation();settings();}},true);
  // Capture grid movement before the continuous drag/manual-thruster handlers.
  el('matrix').addEventListener('pointerdown',event=>{if(active())event.stopImmediatePropagation();},true);
  el('matrix').addEventListener('pointerup',event=>{if(active())event.stopImmediatePropagation();},true);
  document.addEventListener('keydown',event=>{
    if(!active()||event.target.closest('input,select,textarea'))return;
    const value={ArrowUp:0,ArrowDown:1,ArrowLeft:2,ArrowRight:3}[event.key];
    if(value!==undefined){event.preventDefault();event.stopImmediatePropagation();cmd('step',{value});}
  },true);
  document.querySelector('#role-setting-rows [data-role=learner] .role-control-mode').addEventListener('click',event=>{
    if(!active())return;const button=event.target.closest('button');if(!button)return;
    event.preventDefault();event.stopImmediatePropagation();requestedMode=button.dataset.value;send({type:'set_mode',actor:'learner',mode:button.dataset.value});
  },true);
  function active(){return mode==='discrete'&&data;}
  function setValue(id,value,disabled=false){const node=el(id);if(!node)return;if(document.activeElement!==node)node.value=value;node.disabled=disabled;}
  function pressed(node,value){if(node)segments(node,value);}
  function renderSettings(inspected=inspection()){
    if(!active())return;
    for(const [key,id] of Object.entries(fields))setValue(id,(inspected||data.config)[key],Boolean(inspected));
    setValue('training-updates',data.update_budget);setValue('training-time-limit','truncation',true);
    el('training-entropy').checked=(inspected||data.config).entropy_coefficient!==0;el('training-entropy').disabled=false;
    for(const role of ['learner']) for(const id of ['count-min','count-max','speed-ratio-min','speed-ratio-max'])setValue(role+'-'+id,1,true);
    pressed(el('learner-training-mode'),'train');el('learner-training-mode').querySelectorAll('button').forEach(b=>b.disabled=true);
    const controls=document.querySelector('#role-setting-rows [data-role=learner] .role-control-mode');
    pressed(controls,data.sampling==='manual'?'manual':'nn');controls.querySelector('[data-value=rule]').disabled=true;
    el('passage-collision').checked=(inspected||data.config).collision_termination||false;el('passage-collision').disabled=Boolean(inspected);
    el('training-updates').min='0';el('training-updates').title='0 = Unlimited; train until stopped';
    if(inspected)for(const node of el('viewer-settings-section').querySelectorAll('input,select,button'))if(!node.closest('#config-inspection'))node.disabled=true;
  }
  async function settings(){
    if(!active())return false;
    const config={...data.config};for(const [key,id] of Object.entries(fields))config[key]=Number(el(id).value);
    config.collision_termination=el('passage-collision').checked;
    config.entropy_coefficient=el('training-entropy').checked ? .01 : 0;
    return cmd('settings',{config,updates:Number(el('training-updates').value)});
  }
  function toolbar(value){
    if(!active())return;
    rates({...value.goal_reach_viewer,time_scale:data.speed,actual_time_scale:data.actual_time_scale});
    pressed(el('policy-sampling-control'),data.policy_sampling);
    el('evaluation-scene-save').disabled=false;el('evaluation-scene-load').disabled=!data.scene_saved;
    const error=value?.goal_reach_viewer?.command_error||data.deployment_error||data.training.error;
    el('runtime-status').textContent=error||`${data.left?'Left':'Right'} open · ${data.steps}/${data.config.maximum_episode_steps??64} · ${data.done?(data.success?'GOAL':data.outcome==='static_collision'?'COLLISION':'TRUNCATED'):data.sampling==='manual'?'MANUAL · Arrow keys / NN to run':data.running?'RUN':'PAUSED'}`;
    el('runtime-status').classList.toggle('danger',Boolean(error));
    el('runtime-status').title=el('runtime-status').textContent;
    renderSettings();
  }
  function rail(){
    if(!active())return;
    el('selected-actor-section').hidden=false;
    el('selected-actor-section').querySelector('.actor-compact-title').textContent='ATK #0 · State';
    el('selected-state-time').textContent=`${data.done?'ENDED':data.running?'LIVE':'PAUSED'} · Grid · ${data.steps}/${data.config.maximum_episode_steps??64} steps`;
    stateBars(el('selected-telemetry'),[
      {label:'x',value:data.position[0],range:[1,9]},
      {label:'y',value:data.position[1],range:[1,9]},
      {label:'Left open',value:Number(data.left),range:[0,1]},
      {label:'Right open',value:Number(!data.left),range:[0,1]},
    ]);
    const item=(label,value)=>({label,value,range:[0,1]});
    bars(el('selected-vector'),['x','y','Left open','Right open'].map((label,i)=>item(label,data.observation[i])));
    bars(el('selected-policy-output'),['Up','Down','Left','Right'].map((label,i)=>item(label,data.probabilities[i])));
    el('selected-policy-title').textContent='Policy · '+(data.sampling==='manual'?'Manual':'NN');
    el('selected-actor-section').title=data.deployed?`${data.deployed} · U${data.updates}`:'Random';
    source();
  }
  const specs=['policy_loss','value_loss','entropy'].map((key,i)=>({key,label:['Policy loss','Value loss','Entropy'][i],tooltip:key,category:'passage',category_label:'Discrete PPO',scope:'group',unit:'',axis_min:key.includes('loss')?undefined:0,series:[{key,label:'ATK',color_role:'learner'}]}));
  for(const [key,label,tooltip,maximum] of [
    ['approximate_kl','KL · Mean','Mean pre-step approximate KL over accepted optimizer steps; excludes the rejected early-stop check.',null],
    ['max_checked_kl','KL · Max checked','Largest pre-step minibatch KL, including the rejected check. Early-stop threshold = 1.5 × target KL; this is not a post-update KL bound.',null],
    ['clip_fraction','Clip fraction','Mean fraction of sampled likelihood ratios outside the PPO clip interval.',1],
    ['optimizer_steps','Optimizer steps','Actual gradient steps in this update, including reductions from KL early stopping.',null],
  ]) specs.push({key,label,tooltip,category:'passage',category_label:'Discrete PPO',scope:'group',unit:'',axis_min:0,axis_max:maximum,
    accept:record=>Number.isFinite(record[key]),series:[{key,label:'ATK',color_role:'learner'}]});
  specs.push({key:'kl_early_stop_rate',label:'KL early stops · Fixed window',
    tooltip:'Fraction of updates that stopped remaining optimization in a complete, aligned update window. Missing legacy values stay unpublished.',
    category:'passage',category_label:'Discrete PPO',scope:'group',unit:'',axis_min:0,axis_max:1,
    accept:record=>Number.isFinite(record.success_window?.kl_early_stop_rate),
    series:[{key:'kl_early_stop_rate',label:'ATK',color_role:'learner',value:r=>r.success_window.kl_early_stop_rate}]});
  const cohort=r=>r.training_cohorts?.at(-1);
  for(const [key,label,formula,unit] of [
    ['success_rate','Success rate · Train','Successful episodes / completed episodes','%'],
    ['efficiency','Efficiency · Train','16 × successes / complete-episode frames',''],
  ]) specs.push({key:'cohort_'+key,label,
    tooltip:formula+' per side. Episodes are grouped by start window and published only after every outcome is known. X = publication update; hover for start window and n. Delayed training performance, not frozen-policy evaluation. An unsampled side is N/A.',
    category:'passage',category_label:'Discrete PPO',scope:'group',unit,axis_min:0,axis_max:1,
    accept:record=>Boolean(cohort(record)),
    series:['left','right'].map((side,i)=>({key:side+'_'+key,label:i?'Right':'Left',color_role:i?'defender':'learner',
      value:r=>cohort(r)[side+'_'+key],
      detail:r=>{const c=cohort(r);return `starts U${c.start_update}–${c.end_update}, n=${c[side+'_episodes']}, successes=${c[side+'_successes']}, steps=${c[side+'_frames']}`;}}))});
  const minibatch=cfg=>cfg?(cfg.minibatch??cfg.environments*cfg.rollout_steps/(cfg.optimizer_steps/cfg.epochs)):null;
  function groups(){return (data?.catalog||[]).map(row=>{
    const id='passage:'+row.name, cfg=row.config;
    return {scene_mode:'discrete',name:row.name,display_name:passageLabel(cfg),run_group_id:id,started_at:row.started_at,
      source:id,global_update:row.updates,checkpoint_update:row.updates,update_budget:row.update_budget??row.updates,archived:false,status:row.status,error:row.error,
      launch_attempt:row.launch_attempt,update_timing:row.update_timing,evaluation:row.evaluation,
      training_cohort:cohort(row.records?.findLast(r=>cohort(r))||{}),
      settings:{policy_sharing:'role',terminate_on_any_casualty:false,training_settings:cfg},config:cfg,
      records:row.records,parameter_groups:{agent:{members:['learner'],update_index:row.updates}},
      role_bindings:{learner:{key:id+':learner',mode:'train',parameter_group:'agent',source_update:0,architecture_id:data.contract,deploy_reference:id},...Object.fromEntries(['defender','target'].map(role=>[role,{key:id+':'+role,mode:'unavailable',parameter_group:null,deploy_reference:null}]))},
      credit_minibatches:{agent:{members:['learner'],value:minibatch(cfg),update:row.updates}}};
  });}
  function lifecycle(group){const status=group.status;return {liveGroup:group,liveValidated:true,runtimeValid:true,running:status==='running',pending:false,stopping:false,starting:false,completed:false,failed:status==='failed',error:group.error,label:status==='running'?'Running':status==='complete'?'Completed':status==='failed'?'Failed':'Stopped'};}
  function eligibility(reference){
    if(reference==='random_current_architecture')return '';
    const name=reference?.replace(/^passage:/,'');
    if(!data.models.includes(name))return 'Waiting for complete save';
    return data.deployed===name?'Deployed':'';
  }
  async function train(group=null){
    if(group?.status==='running')return cmd('stop',{name:group.name});
    if(!group&&!await settings())return false;
    return cmd('train',{updates:Number(el('training-updates').value),...(group?{name:group.name}:{fresh:true})});
  }
  function deploy(reference){return reference==='random_current_architecture'?cmd('deploy_random'):cmd('load_model',{name:reference.replace(/^passage:/,'')});}
  function draw(context,width,height,palette){
    if(!active())return false;
    context.clearRect(0,0,width,height);context.fillStyle=palette.stage;context.fillRect(0,0,width,height);
    const unit=Math.min(width,height)/12,ox=(width-11*unit)/2,oy=(height-11*unit)/2;
    const css=getComputedStyle(document.documentElement),color=name=>css.getPropertyValue(name).trim();
    const frameColor=color('--canvas-frame'),lineColor=color('--line');
    for(let y=0;y<11;y++)for(let x=0;x<11;x++){
      const wall=x===0||x===10||y===0||y===10||(y===5&&x!==(data.left?1:9));
      context.fillStyle=wall?'#667085':frameColor;context.fillRect(ox+x*unit+1,oy+y*unit+1,unit-2,unit-2);
      if(!wall){context.strokeStyle=lineColor;context.lineWidth=.5;context.strokeRect(ox+x*unit+.5,oy+y*unit+.5,unit-1,unit-1);}
    }
    context.fillStyle='#35bd84';context.fillRect(ox+5.15*unit,oy+1.15*unit,.7*unit,.7*unit);
    context.strokeStyle='#eb606088';context.lineWidth=Math.max(1,unit*.06);context.beginPath();
    data.trail.forEach(([x,y],i)=>context[i?'lineTo':'moveTo'](ox+(x+.5)*unit,oy+(y+.5)*unit));context.stroke();
    context.fillStyle='#ef4444';context.beginPath();context.arc(ox+(data.position[0]+.5)*unit,oy+(data.position[1]+.5)*unit,unit*.22,0,Math.PI*2);context.fill();return true;
  }
  function sync(value){
    const next=value.scene_mode||'continuous',changed=next!==mode;
    if(changed&&next==='discrete'){
      for(const node of document.querySelectorAll('#viewer-settings-section input,#viewer-settings-section select,#toolbar select'))original.set(node,{disabled:node.disabled,readOnly:node.readOnly,min:node.min,title:node.title,value:node.value});
    }
    mode=next;data=value.passage;if(requestedMode===(data?.sampling==='manual'?'manual':'nn'))requestedMode=null;
    if(changed&&mode==='continuous'){
      for(const [node,props] of original)Object.assign(node,props);original.clear();
      document.querySelector('#role-setting-rows [data-role=learner] .role-control-mode [data-value=rule]').hidden=false;
      el('selected-policy-title').textContent='Policy';
    }
    if(changed||!initialized){
      document.body.dataset.sceneMode=mode;el('scene-mode').value=mode;
      for(const node of document.querySelectorAll('.continuous-only,.passage-only')){
        const unavailable=node.classList.contains('continuous-only')?mode==='discrete':mode!=='discrete';
        node.inert=unavailable;node.classList.toggle('scene-unavailable',unavailable);node.title=unavailable?'Unavailable in this scene':'';
      }
      initialized=true;
    }
    const key=JSON.stringify([mode,data?.catalog_revision??data?.catalog,data?.deployed,data?.updates,data?.followed,data?.training.status,data?.config]);
    const rerender=key!==signature;signature=key;
    return changed||rerender;
  }
  return Object.assign(sync,{active,settings,renderSettings,toolbar,rail,groups,specs,lifecycle,eligibility,train,deploy,draw,
    load:async group=>{await deploy('passage:'+group.name);return cmd('settings',{config:group.config,updates:data.update_budget});},
    follow:group=>cmd('follow',{name:data.followed===group.name?null:group.name}),following:()=>data?.followed,
    canStart:()=>Boolean(data),deployed:()=>data?.deployed,deployedUpdate:()=>data?.updates,contract:()=>data?.contract,
    minibatch:()=>minibatch(data?.config)});
}
