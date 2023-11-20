/*************************** 
 * Hybrid_Search_Task Test *
 ***************************/

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color('white'),
  units: 'height',
  waitBlanking: true
});

// store info about the experiment session:
let expName = 'Hybrid_Search_Task';  // from the Builder filename that created this script
let expInfo = {'participant': '', 'age': '', 'gender': ['F', 'M'], 'handedness': ['R', 'L']};

// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
const instr1_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(instr1_loopLoopBegin, instr1_loopLoopScheduler);
flowScheduler.add(instr1_loopLoopScheduler);
flowScheduler.add(instr1_loopLoopEnd);
flowScheduler.add(instrEG1RoutineBegin());
flowScheduler.add(instrEG1RoutineEachFrame());
flowScheduler.add(instrEG1RoutineEnd());
flowScheduler.add(instrEG2RoutineBegin());
flowScheduler.add(instrEG2RoutineEachFrame());
flowScheduler.add(instrEG2RoutineEnd());
const instr2_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(instr2_loopLoopBegin, instr2_loopLoopScheduler);
flowScheduler.add(instr2_loopLoopScheduler);
flowScheduler.add(instr2_loopLoopEnd);
flowScheduler.add(instrSCHegRoutineBegin());
flowScheduler.add(instrSCHegRoutineEachFrame());
flowScheduler.add(instrSCHegRoutineEnd());
flowScheduler.add(instrHintEndRoutineBegin());
flowScheduler.add(instrHintEndRoutineEachFrame());
flowScheduler.add(instrHintEndRoutineEnd());
flowScheduler.add(instrInterRoutineBegin());
flowScheduler.add(instrInterRoutineEachFrame());
flowScheduler.add(instrInterRoutineEnd());
const instr3_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(instr3_loopLoopBegin, instr3_loopLoopScheduler);
flowScheduler.add(instr3_loopLoopScheduler);
flowScheduler.add(instr3_loopLoopEnd);
flowScheduler.add(instrHintExpRoutineBegin());
flowScheduler.add(instrHintExpRoutineEachFrame());
flowScheduler.add(instrHintExpRoutineEnd());
flowScheduler.add(instrEndRoutineBegin());
flowScheduler.add(instrEndRoutineEachFrame());
flowScheduler.add(instrEndRoutineEnd());
flowScheduler.add(pracStartRoutineBegin());
flowScheduler.add(pracStartRoutineEachFrame());
flowScheduler.add(pracStartRoutineEnd());
const practice_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practice_loopLoopBegin, practice_loopLoopScheduler);
flowScheduler.add(practice_loopLoopScheduler);
flowScheduler.add(practice_loopLoopEnd);
flowScheduler.add(expStartRoutineBegin());
flowScheduler.add(expStartRoutineEachFrame());
flowScheduler.add(expStartRoutineEnd());
flowScheduler.add(preExpRoutineBegin());
flowScheduler.add(preExpRoutineEachFrame());
flowScheduler.add(preExpRoutineEnd());
const blockOrdLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(blockOrdLoopBegin, blockOrdLoopScheduler);
flowScheduler.add(blockOrdLoopScheduler);
flowScheduler.add(blockOrdLoopEnd);
flowScheduler.add(goodbyeRoutineBegin());
flowScheduler.add(goodbyeRoutineEachFrame());
flowScheduler.add(goodbyeRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  });

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);


var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2020.2.5';
  expInfo['OS'] = window.navigator.platform;

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  
  return Scheduler.Event.NEXT;
}


var instr1Clock;
var instrImg1;
var instrResp1;
var instrEG1Clock;
var instrImg5;
var instrResp5;
var instrEG2Clock;
var instrImg6;
var instrResp6;
var instr2Clock;
var instrImg3;
var instrResp3;
var instrSCHegClock;
var instrEG;
var egITI;
var instrAnsHint;
var instrEGresp;
var instrHintEndClock;
var instrHint2;
var instrHint2Resp;
var instrInterClock;
var imageITI;
var instrSCHClock;
var instrImgSCH;
var instrITI;
var instrRespSCH;
var instrHintExpClock;
var instrHint3;
var instrHint3Resp;
var instrEndClock;
var instr10;
var instrResp10;
var pracStartClock;
var pStartHint;
var pITI;
var pTBRctrlClock;
var jdgTestFileClock;
var pStudyHintClock;
var psHint;
var psFix;
var pStudyTrialsClock;
var pStudyImg;
var pStdITI;
var pTestHintClock;
var ptHint;
var ptFix;
var pTestTrialsClock;
var pTestImg;
var pTestResp;
var pTextITIClock;
var pfbkTest;
var pTestITI;
var pACCcountClock;
var endHintClock;
var endHintImg;
var endConResp;
var pSearchHintClock;
var sHint;
var sFix;
var pSearchTrialsClock;
var psImg;
var psITI;
var psResp;
var pACCpresenClock;
var pACCpresen1;
var pACCpresen2;
var expStartClock;
var formalExp;
var expStaResp;
var preExpClock;
var subCountBeginClock;
var stimCountBeginClock;
var inAllStimClock;
var subCountPlusClock;
var distrCtrlClock;
var altStimCountBeginClock;
var inAltDistrClock;
var distrCountBeginClock;
var inBothDistrClock;
var listMergeClock;
var TBRLoopCtrlClock;
var studyHintClock;
var stHint;
var stFix;
var studyTrialsClock;
var studyImg;
var studyITI;
var testHintClock;
var ttHint;
var ttFix;
var testTrialsClock;
var testImg;
var testResp;
var test_itiClock;
var fbkTest;
var testITI;
var TBRcountPlusClock;
var expENDClock;
var expENDtest;
var searchHintClock;
var schHint;
var schFix;
var searchTrialsClock;
var schImg;
var schITI;
var schResp;
var schACCpresenClock;
var ACCpresen1;
var ACCpresen2;
var ACCpresen3;
var ACCpresen4;
var ACCpresen5;
var ACCpresen6;
var brkResp;
var goodbyeClock;
var gdy;
var globalClock;
var routineTimer;
function experimentInit() {
  // Initialize components for Routine "instr1"
  instr1Clock = new util.Clock();
  instrImg1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrImg1', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrResp1 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrEG1"
  instrEG1Clock = new util.Clock();
  instrImg5 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrImg5', units : undefined, 
    image : 'Instructions/instr5.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrResp5 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrEG2"
  instrEG2Clock = new util.Clock();
  instrImg6 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrImg6', units : undefined, 
    image : 'Instructions/instr6.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrResp6 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instr2"
  instr2Clock = new util.Clock();
  instrImg3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrImg3', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrResp3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrSCHeg"
  instrSCHegClock = new util.Clock();
  instrEG = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrEG', units : undefined, 
    image : 'Instructions/s1.jpg', mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  egITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'egITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  instrAnsHint = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrAnsHint', units : undefined, 
    image : 'Instructions/instrHint1.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  instrEGresp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrHintEnd"
  instrHintEndClock = new util.Clock();
  instrHint2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrHint2', units : undefined, 
    image : 'Instructions/instrHint2.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrHint2Resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrInter"
  instrInterClock = new util.Clock();
  imageITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'imageITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  // Initialize components for Routine "instrSCH"
  instrSCHClock = new util.Clock();
  instrImgSCH = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrImgSCH', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  instrRespSCH = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrHintExp"
  instrHintExpClock = new util.Clock();
  instrHint3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instrHint3', units : undefined, 
    image : 'Instructions/instrHint3.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrHint3Resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instrEnd"
  instrEndClock = new util.Clock();
  instr10 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'instr10', units : undefined, 
    image : 'Instructions/instr10.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  instrResp10 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pracStart"
  pracStartClock = new util.Clock();
  pStartHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'pStartHint',
    text: 'Practice',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  pITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  // Initialize components for Routine "pTBRctrl"
  pTBRctrlClock = new util.Clock();
  // Initialize components for Routine "jdgTestFile"
  jdgTestFileClock = new util.Clock();
  // Initialize components for Routine "pStudyHint"
  pStudyHintClock = new util.Clock();
  psHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'psHint',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  psFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'psFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "pStudyTrials"
  pStudyTrialsClock = new util.Clock();
  pStudyImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pStudyImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  pStdITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pStdITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  // Initialize components for Routine "pTestHint"
  pTestHintClock = new util.Clock();
  ptHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'ptHint',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  ptFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'ptFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "pTestTrials"
  pTestTrialsClock = new util.Clock();
  pTestImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pTestImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  pTestResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pTextITI"
  pTextITIClock = new util.Clock();
  pfbkTest = new visual.TextStim({
    win: psychoJS.window,
    name: 'pfbkTest',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: -1.0 
  });
  
  pTestITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pTestITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "pACCcount"
  pACCcountClock = new util.Clock();
  // Initialize components for Routine "endHint"
  endHintClock = new util.Clock();
  endHintImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'endHintImg', units : undefined, 
    image : 'Instructions/endHint.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  endConResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pSearchHint"
  pSearchHintClock = new util.Clock();
  sHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'sHint',
    text: 'Searching',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  sFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'sFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  // Initialize components for Routine "pSearchTrials"
  pSearchTrialsClock = new util.Clock();
  psImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'psImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  psITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'psITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  psResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pACCpresen"
  pACCpresenClock = new util.Clock();
  pACCpresen1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'pACCpresen1',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.05], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  pACCpresen2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'pACCpresen2',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.05)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  // Initialize components for Routine "expStart"
  expStartClock = new util.Clock();
  formalExp = new visual.ImageStim({
    win : psychoJS.window,
    name : 'formalExp', units : undefined, 
    image : 'Instructions/instr_exp.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  expStaResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "preExp"
  preExpClock = new util.Clock();
  // Initialize components for Routine "subCountBegin"
  subCountBeginClock = new util.Clock();
  // Initialize components for Routine "stimCountBegin"
  stimCountBeginClock = new util.Clock();
  // Initialize components for Routine "inAllStim"
  inAllStimClock = new util.Clock();
  // Initialize components for Routine "subCountPlus"
  subCountPlusClock = new util.Clock();
  // Initialize components for Routine "distrCtrl"
  distrCtrlClock = new util.Clock();
  // Initialize components for Routine "altStimCountBegin"
  altStimCountBeginClock = new util.Clock();
  // Initialize components for Routine "inAltDistr"
  inAltDistrClock = new util.Clock();
  // Initialize components for Routine "distrCountBegin"
  distrCountBeginClock = new util.Clock();
  // Initialize components for Routine "inBothDistr"
  inBothDistrClock = new util.Clock();
  // Initialize components for Routine "listMerge"
  listMergeClock = new util.Clock();
  // Initialize components for Routine "TBRLoopCtrl"
  TBRLoopCtrlClock = new util.Clock();
  // Initialize components for Routine "studyHint"
  studyHintClock = new util.Clock();
  stHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'stHint',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  stFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'stFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "studyTrials"
  studyTrialsClock = new util.Clock();
  studyImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'studyImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  studyITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'studyITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "testHint"
  testHintClock = new util.Clock();
  ttHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'ttHint',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  ttFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'ttFix',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "testTrials"
  testTrialsClock = new util.Clock();
  testImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'testImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  testResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "test_iti"
  test_itiClock = new util.Clock();
  fbkTest = new visual.TextStim({
    win: psychoJS.window,
    name: 'fbkTest',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  testITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'testITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  // Initialize components for Routine "TBRcountPlus"
  TBRcountPlusClock = new util.Clock();
  // Initialize components for Routine "expEND"
  expENDClock = new util.Clock();
  expENDtest = new visual.TextStim({
    win: psychoJS.window,
    name: 'expENDtest',
    text: 'Your accuracy is too low, so we have to finish the experiment.\nPlease enter Esc to exit full screen mode and then close the browser.\nThank you for your participation.\nHave a good day.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Initialize components for Routine "searchHint"
  searchHintClock = new util.Clock();
  schHint = new visual.TextStim({
    win: psychoJS.window,
    name: 'schHint',
    text: 'Searching',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  schFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'schFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  // Initialize components for Routine "searchTrials"
  searchTrialsClock = new util.Clock();
  schImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'schImg', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  schITI = new visual.ImageStim({
    win : psychoJS.window,
    name : 'schITI', units : undefined, 
    image : 'Instructions/ITI.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  schResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "schACCpresen"
  schACCpresenClock = new util.Clock();
  ACCpresen1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen1',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.2], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -1.0 
  });
  
  ACCpresen2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen2',
    text: 'default text',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.1], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  ACCpresen3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen3',
    text: '-----  Break (within 5 min) -----',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.2)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -3.0 
  });
  
  ACCpresen4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen4',
    text: 'Please press',
    font: 'Arial',
    units: undefined, 
    pos: [(- 0.3), (- 0.3)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -4.0 
  });
  
  ACCpresen5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen5',
    text: 'SPACE',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.3)], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('red'),  opacity: 1,
    depth: -5.0 
  });
  
  ACCpresen6 = new visual.TextStim({
    win: psychoJS.window,
    name: 'ACCpresen6',
    text: 'to continue.',
    font: 'Arial',
    units: undefined, 
    pos: [0.3, (- 0.3)], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: -6.0 
  });
  
  brkResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "goodbye"
  goodbyeClock = new util.Clock();
  gdy = new visual.TextStim({
    win: psychoJS.window,
    name: 'gdy',
    text: 'All done!\nYour experiment will be completely finished after a few seconds.\nHave a nice day.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.06,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var instr1_loop;
var currentLoop;
function instr1_loopLoopBegin(instr1_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  instr1_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'Instructions/instrP1.xlsx',
    seed: undefined, name: 'instr1_loop'
  });
  psychoJS.experiment.addLoop(instr1_loop); // add the loop to the experiment
  currentLoop = instr1_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  instr1_loop.forEach(function() {
    const snapshot = instr1_loop.getSnapshot();

    instr1_loopLoopScheduler.add(importConditions(snapshot));
    instr1_loopLoopScheduler.add(instr1RoutineBegin(snapshot));
    instr1_loopLoopScheduler.add(instr1RoutineEachFrame(snapshot));
    instr1_loopLoopScheduler.add(instr1RoutineEnd(snapshot));
    instr1_loopLoopScheduler.add(endLoopIteration(instr1_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function instr1_loopLoopEnd() {
  psychoJS.experiment.removeLoop(instr1_loop);

  return Scheduler.Event.NEXT;
}


var instr2_loop;
function instr2_loopLoopBegin(instr2_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  instr2_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'Instructions/instrP2.xlsx',
    seed: undefined, name: 'instr2_loop'
  });
  psychoJS.experiment.addLoop(instr2_loop); // add the loop to the experiment
  currentLoop = instr2_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  instr2_loop.forEach(function() {
    const snapshot = instr2_loop.getSnapshot();

    instr2_loopLoopScheduler.add(importConditions(snapshot));
    instr2_loopLoopScheduler.add(instr2RoutineBegin(snapshot));
    instr2_loopLoopScheduler.add(instr2RoutineEachFrame(snapshot));
    instr2_loopLoopScheduler.add(instr2RoutineEnd(snapshot));
    instr2_loopLoopScheduler.add(endLoopIteration(instr2_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function instr2_loopLoopEnd() {
  psychoJS.experiment.removeLoop(instr2_loop);

  return Scheduler.Event.NEXT;
}


var instr3_loop;
function instr3_loopLoopBegin(instr3_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  instr3_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'Instructions/instrP3.xlsx',
    seed: undefined, name: 'instr3_loop'
  });
  psychoJS.experiment.addLoop(instr3_loop); // add the loop to the experiment
  currentLoop = instr3_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  instr3_loop.forEach(function() {
    const snapshot = instr3_loop.getSnapshot();

    instr3_loopLoopScheduler.add(importConditions(snapshot));
    instr3_loopLoopScheduler.add(instrSCHRoutineBegin(snapshot));
    instr3_loopLoopScheduler.add(instrSCHRoutineEachFrame(snapshot));
    instr3_loopLoopScheduler.add(instrSCHRoutineEnd(snapshot));
    instr3_loopLoopScheduler.add(endLoopIteration(instr3_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function instr3_loopLoopEnd() {
  psychoJS.experiment.removeLoop(instr3_loop);

  return Scheduler.Event.NEXT;
}


var practice_loop;
function practice_loopLoopBegin(practice_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  practice_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'Practice/pstList.xlsx',
    seed: undefined, name: 'practice_loop'
  });
  psychoJS.experiment.addLoop(practice_loop); // add the loop to the experiment
  currentLoop = practice_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  practice_loop.forEach(function() {
    const snapshot = practice_loop.getSnapshot();

    practice_loopLoopScheduler.add(importConditions(snapshot));
    practice_loopLoopScheduler.add(pTBRctrlRoutineBegin(snapshot));
    practice_loopLoopScheduler.add(pTBRctrlRoutineEachFrame(snapshot));
    practice_loopLoopScheduler.add(pTBRctrlRoutineEnd(snapshot));
    const pTBR_loopLoopScheduler = new Scheduler(psychoJS);
    practice_loopLoopScheduler.add(pTBR_loopLoopBegin, pTBR_loopLoopScheduler);
    practice_loopLoopScheduler.add(pTBR_loopLoopScheduler);
    practice_loopLoopScheduler.add(pTBR_loopLoopEnd);
    practice_loopLoopScheduler.add(pSearchHintRoutineBegin(snapshot));
    practice_loopLoopScheduler.add(pSearchHintRoutineEachFrame(snapshot));
    practice_loopLoopScheduler.add(pSearchHintRoutineEnd(snapshot));
    const pSearch_loopLoopScheduler = new Scheduler(psychoJS);
    practice_loopLoopScheduler.add(pSearch_loopLoopBegin, pSearch_loopLoopScheduler);
    practice_loopLoopScheduler.add(pSearch_loopLoopScheduler);
    practice_loopLoopScheduler.add(pSearch_loopLoopEnd);
    practice_loopLoopScheduler.add(pACCpresenRoutineBegin(snapshot));
    practice_loopLoopScheduler.add(pACCpresenRoutineEachFrame(snapshot));
    practice_loopLoopScheduler.add(pACCpresenRoutineEnd(snapshot));
    practice_loopLoopScheduler.add(endLoopIteration(practice_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var pTBR_loop;
function pTBR_loopLoopBegin(pTBR_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  pTBR_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 5, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'pTBR_loop'
  });
  psychoJS.experiment.addLoop(pTBR_loop); // add the loop to the experiment
  currentLoop = pTBR_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  pTBR_loop.forEach(function() {
    const snapshot = pTBR_loop.getSnapshot();

    pTBR_loopLoopScheduler.add(importConditions(snapshot));
    pTBR_loopLoopScheduler.add(jdgTestFileRoutineBegin(snapshot));
    pTBR_loopLoopScheduler.add(jdgTestFileRoutineEachFrame(snapshot));
    pTBR_loopLoopScheduler.add(jdgTestFileRoutineEnd(snapshot));
    const psPresen_loopLoopScheduler = new Scheduler(psychoJS);
    pTBR_loopLoopScheduler.add(psPresen_loopLoopBegin, psPresen_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(psPresen_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(psPresen_loopLoopEnd);
    const ptHint_loopLoopScheduler = new Scheduler(psychoJS);
    pTBR_loopLoopScheduler.add(ptHint_loopLoopBegin, ptHint_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(ptHint_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(ptHint_loopLoopEnd);
    const pTest_loopLoopScheduler = new Scheduler(psychoJS);
    pTBR_loopLoopScheduler.add(pTest_loopLoopBegin, pTest_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(pTest_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(pTest_loopLoopEnd);
    pTBR_loopLoopScheduler.add(pACCcountRoutineBegin(snapshot));
    pTBR_loopLoopScheduler.add(pACCcountRoutineEachFrame(snapshot));
    pTBR_loopLoopScheduler.add(pACCcountRoutineEnd(snapshot));
    const endHint_loopLoopScheduler = new Scheduler(psychoJS);
    pTBR_loopLoopScheduler.add(endHint_loopLoopBegin, endHint_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(endHint_loopLoopScheduler);
    pTBR_loopLoopScheduler.add(endHint_loopLoopEnd);
    pTBR_loopLoopScheduler.add(endLoopIteration(pTBR_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var psPresen_loop;
function psPresen_loopLoopBegin(psPresen_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  psPresen_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: pstLoopCtrl, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'psPresen_loop'
  });
  psychoJS.experiment.addLoop(psPresen_loop); // add the loop to the experiment
  currentLoop = psPresen_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  psPresen_loop.forEach(function() {
    const snapshot = psPresen_loop.getSnapshot();

    psPresen_loopLoopScheduler.add(importConditions(snapshot));
    psPresen_loopLoopScheduler.add(pStudyHintRoutineBegin(snapshot));
    psPresen_loopLoopScheduler.add(pStudyHintRoutineEachFrame(snapshot));
    psPresen_loopLoopScheduler.add(pStudyHintRoutineEnd(snapshot));
    const pStudy_loopLoopScheduler = new Scheduler(psychoJS);
    psPresen_loopLoopScheduler.add(pStudy_loopLoopBegin, pStudy_loopLoopScheduler);
    psPresen_loopLoopScheduler.add(pStudy_loopLoopScheduler);
    psPresen_loopLoopScheduler.add(pStudy_loopLoopEnd);
    psPresen_loopLoopScheduler.add(endLoopIteration(psPresen_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var pStudy_loop;
function pStudy_loopLoopBegin(pStudy_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  pStudy_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: pStudyOrd,
    seed: undefined, name: 'pStudy_loop'
  });
  psychoJS.experiment.addLoop(pStudy_loop); // add the loop to the experiment
  currentLoop = pStudy_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  pStudy_loop.forEach(function() {
    const snapshot = pStudy_loop.getSnapshot();

    pStudy_loopLoopScheduler.add(importConditions(snapshot));
    pStudy_loopLoopScheduler.add(pStudyTrialsRoutineBegin(snapshot));
    pStudy_loopLoopScheduler.add(pStudyTrialsRoutineEachFrame(snapshot));
    pStudy_loopLoopScheduler.add(pStudyTrialsRoutineEnd(snapshot));
    pStudy_loopLoopScheduler.add(endLoopIteration(pStudy_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function pStudy_loopLoopEnd() {
  psychoJS.experiment.removeLoop(pStudy_loop);

  return Scheduler.Event.NEXT;
}


function psPresen_loopLoopEnd() {
  psychoJS.experiment.removeLoop(psPresen_loop);

  return Scheduler.Event.NEXT;
}


var ptHint_loop;
function ptHint_loopLoopBegin(ptHint_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  ptHint_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: pthLoopCtrl , method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'ptHint_loop'
  });
  psychoJS.experiment.addLoop(ptHint_loop); // add the loop to the experiment
  currentLoop = ptHint_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  ptHint_loop.forEach(function() {
    const snapshot = ptHint_loop.getSnapshot();

    ptHint_loopLoopScheduler.add(importConditions(snapshot));
    ptHint_loopLoopScheduler.add(pTestHintRoutineBegin(snapshot));
    ptHint_loopLoopScheduler.add(pTestHintRoutineEachFrame(snapshot));
    ptHint_loopLoopScheduler.add(pTestHintRoutineEnd(snapshot));
    ptHint_loopLoopScheduler.add(endLoopIteration(ptHint_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function ptHint_loopLoopEnd() {
  psychoJS.experiment.removeLoop(ptHint_loop);

  return Scheduler.Event.NEXT;
}


var pTest_loop;
function pTest_loopLoopBegin(pTest_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  pTest_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: ptestLoopCtrl, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: pTestFile,
    seed: undefined, name: 'pTest_loop'
  });
  psychoJS.experiment.addLoop(pTest_loop); // add the loop to the experiment
  currentLoop = pTest_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  pTest_loop.forEach(function() {
    const snapshot = pTest_loop.getSnapshot();

    pTest_loopLoopScheduler.add(importConditions(snapshot));
    pTest_loopLoopScheduler.add(pTestTrialsRoutineBegin(snapshot));
    pTest_loopLoopScheduler.add(pTestTrialsRoutineEachFrame(snapshot));
    pTest_loopLoopScheduler.add(pTestTrialsRoutineEnd(snapshot));
    pTest_loopLoopScheduler.add(pTextITIRoutineBegin(snapshot));
    pTest_loopLoopScheduler.add(pTextITIRoutineEachFrame(snapshot));
    pTest_loopLoopScheduler.add(pTextITIRoutineEnd(snapshot));
    pTest_loopLoopScheduler.add(endLoopIteration(pTest_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function pTest_loopLoopEnd() {
  psychoJS.experiment.removeLoop(pTest_loop);

  return Scheduler.Event.NEXT;
}


var endHint_loop;
function endHint_loopLoopBegin(endHint_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  endHint_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: pendExpTag, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'endHint_loop'
  });
  psychoJS.experiment.addLoop(endHint_loop); // add the loop to the experiment
  currentLoop = endHint_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  endHint_loop.forEach(function() {
    const snapshot = endHint_loop.getSnapshot();

    endHint_loopLoopScheduler.add(importConditions(snapshot));
    endHint_loopLoopScheduler.add(endHintRoutineBegin(snapshot));
    endHint_loopLoopScheduler.add(endHintRoutineEachFrame(snapshot));
    endHint_loopLoopScheduler.add(endHintRoutineEnd(snapshot));
    endHint_loopLoopScheduler.add(endLoopIteration(endHint_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function endHint_loopLoopEnd() {
  psychoJS.experiment.removeLoop(endHint_loop);

  return Scheduler.Event.NEXT;
}


function pTBR_loopLoopEnd() {
  psychoJS.experiment.removeLoop(pTBR_loop);

  return Scheduler.Event.NEXT;
}


var pSearch_loop;
function pSearch_loopLoopBegin(pSearch_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  pSearch_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: pSearch,
    seed: undefined, name: 'pSearch_loop'
  });
  psychoJS.experiment.addLoop(pSearch_loop); // add the loop to the experiment
  currentLoop = pSearch_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  pSearch_loop.forEach(function() {
    const snapshot = pSearch_loop.getSnapshot();

    pSearch_loopLoopScheduler.add(importConditions(snapshot));
    pSearch_loopLoopScheduler.add(pSearchTrialsRoutineBegin(snapshot));
    pSearch_loopLoopScheduler.add(pSearchTrialsRoutineEachFrame(snapshot));
    pSearch_loopLoopScheduler.add(pSearchTrialsRoutineEnd(snapshot));
    pSearch_loopLoopScheduler.add(endLoopIteration(pSearch_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function pSearch_loopLoopEnd() {
  psychoJS.experiment.removeLoop(pSearch_loop);

  return Scheduler.Event.NEXT;
}


function practice_loopLoopEnd() {
  psychoJS.experiment.removeLoop(practice_loop);

  return Scheduler.Event.NEXT;
}


var blockOrd;
function blockOrdLoopBegin(blockOrdLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  blockOrd = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'StimList/setOrdObj.xlsx',
    seed: undefined, name: 'blockOrd'
  });
  psychoJS.experiment.addLoop(blockOrd); // add the loop to the experiment
  currentLoop = blockOrd;  // we're now the current loop

  // Schedule all the trials in the trialList:
  blockOrd.forEach(function() {
    const snapshot = blockOrd.getSnapshot();

    blockOrdLoopScheduler.add(importConditions(snapshot));
    blockOrdLoopScheduler.add(subCountBeginRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(subCountBeginRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(subCountBeginRoutineEnd(snapshot));
    const sub30_loopLoopScheduler = new Scheduler(psychoJS);
    blockOrdLoopScheduler.add(sub30_loopLoopBegin, sub30_loopLoopScheduler);
    blockOrdLoopScheduler.add(sub30_loopLoopScheduler);
    blockOrdLoopScheduler.add(sub30_loopLoopEnd);
    blockOrdLoopScheduler.add(distrCtrlRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(distrCtrlRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(distrCtrlRoutineEnd(snapshot));
    const distr_loopLoopScheduler = new Scheduler(psychoJS);
    blockOrdLoopScheduler.add(distr_loopLoopBegin, distr_loopLoopScheduler);
    blockOrdLoopScheduler.add(distr_loopLoopScheduler);
    blockOrdLoopScheduler.add(distr_loopLoopEnd);
    blockOrdLoopScheduler.add(distrCountBeginRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(distrCountBeginRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(distrCountBeginRoutineEnd(snapshot));
    const chooseDistr_loopLoopScheduler = new Scheduler(psychoJS);
    blockOrdLoopScheduler.add(chooseDistr_loopLoopBegin, chooseDistr_loopLoopScheduler);
    blockOrdLoopScheduler.add(chooseDistr_loopLoopScheduler);
    blockOrdLoopScheduler.add(chooseDistr_loopLoopEnd);
    blockOrdLoopScheduler.add(listMergeRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(listMergeRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(listMergeRoutineEnd(snapshot));
    const TBR_loopLoopScheduler = new Scheduler(psychoJS);
    blockOrdLoopScheduler.add(TBR_loopLoopBegin, TBR_loopLoopScheduler);
    blockOrdLoopScheduler.add(TBR_loopLoopScheduler);
    blockOrdLoopScheduler.add(TBR_loopLoopEnd);
    blockOrdLoopScheduler.add(searchHintRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(searchHintRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(searchHintRoutineEnd(snapshot));
    const sch_loopLoopScheduler = new Scheduler(psychoJS);
    blockOrdLoopScheduler.add(sch_loopLoopBegin, sch_loopLoopScheduler);
    blockOrdLoopScheduler.add(sch_loopLoopScheduler);
    blockOrdLoopScheduler.add(sch_loopLoopEnd);
    blockOrdLoopScheduler.add(schACCpresenRoutineBegin(snapshot));
    blockOrdLoopScheduler.add(schACCpresenRoutineEachFrame(snapshot));
    blockOrdLoopScheduler.add(schACCpresenRoutineEnd(snapshot));
    blockOrdLoopScheduler.add(endLoopIteration(blockOrdLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var sub30_loop;
function sub30_loopLoopBegin(sub30_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  sub30_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: stimCateList,
    seed: undefined, name: 'sub30_loop'
  });
  psychoJS.experiment.addLoop(sub30_loop); // add the loop to the experiment
  currentLoop = sub30_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  sub30_loop.forEach(function() {
    const snapshot = sub30_loop.getSnapshot();

    sub30_loopLoopScheduler.add(importConditions(snapshot));
    sub30_loopLoopScheduler.add(stimCountBeginRoutineBegin(snapshot));
    sub30_loopLoopScheduler.add(stimCountBeginRoutineEachFrame(snapshot));
    sub30_loopLoopScheduler.add(stimCountBeginRoutineEnd(snapshot));
    const stim17_loopLoopScheduler = new Scheduler(psychoJS);
    sub30_loopLoopScheduler.add(stim17_loopLoopBegin, stim17_loopLoopScheduler);
    sub30_loopLoopScheduler.add(stim17_loopLoopScheduler);
    sub30_loopLoopScheduler.add(stim17_loopLoopEnd);
    sub30_loopLoopScheduler.add(subCountPlusRoutineBegin(snapshot));
    sub30_loopLoopScheduler.add(subCountPlusRoutineEachFrame(snapshot));
    sub30_loopLoopScheduler.add(subCountPlusRoutineEnd(snapshot));
    sub30_loopLoopScheduler.add(endLoopIteration(sub30_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var stim17_loop;
function stim17_loopLoopBegin(stim17_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  stim17_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: stimSubCate,
    seed: undefined, name: 'stim17_loop'
  });
  psychoJS.experiment.addLoop(stim17_loop); // add the loop to the experiment
  currentLoop = stim17_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  stim17_loop.forEach(function() {
    const snapshot = stim17_loop.getSnapshot();

    stim17_loopLoopScheduler.add(importConditions(snapshot));
    stim17_loopLoopScheduler.add(inAllStimRoutineBegin(snapshot));
    stim17_loopLoopScheduler.add(inAllStimRoutineEachFrame(snapshot));
    stim17_loopLoopScheduler.add(inAllStimRoutineEnd(snapshot));
    stim17_loopLoopScheduler.add(endLoopIteration(stim17_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function stim17_loopLoopEnd() {
  psychoJS.experiment.removeLoop(stim17_loop);

  return Scheduler.Event.NEXT;
}


function sub30_loopLoopEnd() {
  psychoJS.experiment.removeLoop(sub30_loop);

  return Scheduler.Event.NEXT;
}


var distr_loop;
function distr_loopLoopBegin(distr_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  distr_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'StimList/animCate.xls',
    seed: undefined, name: 'distr_loop'
  });
  psychoJS.experiment.addLoop(distr_loop); // add the loop to the experiment
  currentLoop = distr_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  distr_loop.forEach(function() {
    const snapshot = distr_loop.getSnapshot();

    distr_loopLoopScheduler.add(importConditions(snapshot));
    distr_loopLoopScheduler.add(altStimCountBeginRoutineBegin(snapshot));
    distr_loopLoopScheduler.add(altStimCountBeginRoutineEachFrame(snapshot));
    distr_loopLoopScheduler.add(altStimCountBeginRoutineEnd(snapshot));
    const altDistrStim_loopLoopScheduler = new Scheduler(psychoJS);
    distr_loopLoopScheduler.add(altDistrStim_loopLoopBegin, altDistrStim_loopLoopScheduler);
    distr_loopLoopScheduler.add(altDistrStim_loopLoopScheduler);
    distr_loopLoopScheduler.add(altDistrStim_loopLoopEnd);
    distr_loopLoopScheduler.add(endLoopIteration(distr_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var altDistrStim_loop;
function altDistrStim_loopLoopBegin(altDistrStim_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  altDistrStim_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: stimSubCate,
    seed: undefined, name: 'altDistrStim_loop'
  });
  psychoJS.experiment.addLoop(altDistrStim_loop); // add the loop to the experiment
  currentLoop = altDistrStim_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  altDistrStim_loop.forEach(function() {
    const snapshot = altDistrStim_loop.getSnapshot();

    altDistrStim_loopLoopScheduler.add(importConditions(snapshot));
    altDistrStim_loopLoopScheduler.add(inAltDistrRoutineBegin(snapshot));
    altDistrStim_loopLoopScheduler.add(inAltDistrRoutineEachFrame(snapshot));
    altDistrStim_loopLoopScheduler.add(inAltDistrRoutineEnd(snapshot));
    altDistrStim_loopLoopScheduler.add(endLoopIteration(altDistrStim_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function altDistrStim_loopLoopEnd() {
  psychoJS.experiment.removeLoop(altDistrStim_loop);

  return Scheduler.Event.NEXT;
}


function distr_loopLoopEnd() {
  psychoJS.experiment.removeLoop(distr_loop);

  return Scheduler.Event.NEXT;
}


var chooseDistr_loop;
function chooseDistr_loopLoopBegin(chooseDistr_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  chooseDistr_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'StimList/distrIndxList.xlsx',
    seed: undefined, name: 'chooseDistr_loop'
  });
  psychoJS.experiment.addLoop(chooseDistr_loop); // add the loop to the experiment
  currentLoop = chooseDistr_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  chooseDistr_loop.forEach(function() {
    const snapshot = chooseDistr_loop.getSnapshot();

    chooseDistr_loopLoopScheduler.add(importConditions(snapshot));
    chooseDistr_loopLoopScheduler.add(inBothDistrRoutineBegin(snapshot));
    chooseDistr_loopLoopScheduler.add(inBothDistrRoutineEachFrame(snapshot));
    chooseDistr_loopLoopScheduler.add(inBothDistrRoutineEnd(snapshot));
    chooseDistr_loopLoopScheduler.add(endLoopIteration(chooseDistr_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function chooseDistr_loopLoopEnd() {
  psychoJS.experiment.removeLoop(chooseDistr_loop);

  return Scheduler.Event.NEXT;
}


var TBR_loop;
function TBR_loopLoopBegin(TBR_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  TBR_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 5, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'TBR_loop'
  });
  psychoJS.experiment.addLoop(TBR_loop); // add the loop to the experiment
  currentLoop = TBR_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  TBR_loop.forEach(function() {
    const snapshot = TBR_loop.getSnapshot();

    TBR_loopLoopScheduler.add(importConditions(snapshot));
    TBR_loopLoopScheduler.add(TBRLoopCtrlRoutineBegin(snapshot));
    TBR_loopLoopScheduler.add(TBRLoopCtrlRoutineEachFrame(snapshot));
    TBR_loopLoopScheduler.add(TBRLoopCtrlRoutineEnd(snapshot));
    const study_presenLoopScheduler = new Scheduler(psychoJS);
    TBR_loopLoopScheduler.add(study_presenLoopBegin, study_presenLoopScheduler);
    TBR_loopLoopScheduler.add(study_presenLoopScheduler);
    TBR_loopLoopScheduler.add(study_presenLoopEnd);
    const testHint_presenLoopScheduler = new Scheduler(psychoJS);
    TBR_loopLoopScheduler.add(testHint_presenLoopBegin, testHint_presenLoopScheduler);
    TBR_loopLoopScheduler.add(testHint_presenLoopScheduler);
    TBR_loopLoopScheduler.add(testHint_presenLoopEnd);
    const test_loopLoopScheduler = new Scheduler(psychoJS);
    TBR_loopLoopScheduler.add(test_loopLoopBegin, test_loopLoopScheduler);
    TBR_loopLoopScheduler.add(test_loopLoopScheduler);
    TBR_loopLoopScheduler.add(test_loopLoopEnd);
    TBR_loopLoopScheduler.add(TBRcountPlusRoutineBegin(snapshot));
    TBR_loopLoopScheduler.add(TBRcountPlusRoutineEachFrame(snapshot));
    TBR_loopLoopScheduler.add(TBRcountPlusRoutineEnd(snapshot));
    const expCtrl_loopLoopScheduler = new Scheduler(psychoJS);
    TBR_loopLoopScheduler.add(expCtrl_loopLoopBegin, expCtrl_loopLoopScheduler);
    TBR_loopLoopScheduler.add(expCtrl_loopLoopScheduler);
    TBR_loopLoopScheduler.add(expCtrl_loopLoopEnd);
    TBR_loopLoopScheduler.add(endLoopIteration(TBR_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var study_presen;
function study_presenLoopBegin(study_presenLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  study_presen = new TrialHandler({
    psychoJS: psychoJS,
    nReps: stLoopCtrl, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'study_presen'
  });
  psychoJS.experiment.addLoop(study_presen); // add the loop to the experiment
  currentLoop = study_presen;  // we're now the current loop

  // Schedule all the trials in the trialList:
  study_presen.forEach(function() {
    const snapshot = study_presen.getSnapshot();

    study_presenLoopScheduler.add(importConditions(snapshot));
    study_presenLoopScheduler.add(studyHintRoutineBegin(snapshot));
    study_presenLoopScheduler.add(studyHintRoutineEachFrame(snapshot));
    study_presenLoopScheduler.add(studyHintRoutineEnd(snapshot));
    const study_loopLoopScheduler = new Scheduler(psychoJS);
    study_presenLoopScheduler.add(study_loopLoopBegin, study_loopLoopScheduler);
    study_presenLoopScheduler.add(study_loopLoopScheduler);
    study_presenLoopScheduler.add(study_loopLoopEnd);
    study_presenLoopScheduler.add(endLoopIteration(study_presenLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


var study_loop;
function study_loopLoopBegin(study_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  study_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: setsize, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'study_loop'
  });
  psychoJS.experiment.addLoop(study_loop); // add the loop to the experiment
  currentLoop = study_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  study_loop.forEach(function() {
    const snapshot = study_loop.getSnapshot();

    study_loopLoopScheduler.add(importConditions(snapshot));
    study_loopLoopScheduler.add(studyTrialsRoutineBegin(snapshot));
    study_loopLoopScheduler.add(studyTrialsRoutineEachFrame(snapshot));
    study_loopLoopScheduler.add(studyTrialsRoutineEnd(snapshot));
    study_loopLoopScheduler.add(endLoopIteration(study_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function study_loopLoopEnd() {
  psychoJS.experiment.removeLoop(study_loop);

  return Scheduler.Event.NEXT;
}


function study_presenLoopEnd() {
  psychoJS.experiment.removeLoop(study_presen);

  return Scheduler.Event.NEXT;
}


var testHint_presen;
function testHint_presenLoopBegin(testHint_presenLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  testHint_presen = new TrialHandler({
    psychoJS: psychoJS,
    nReps: thLoopCtrl, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'testHint_presen'
  });
  psychoJS.experiment.addLoop(testHint_presen); // add the loop to the experiment
  currentLoop = testHint_presen;  // we're now the current loop

  // Schedule all the trials in the trialList:
  testHint_presen.forEach(function() {
    const snapshot = testHint_presen.getSnapshot();

    testHint_presenLoopScheduler.add(importConditions(snapshot));
    testHint_presenLoopScheduler.add(testHintRoutineBegin(snapshot));
    testHint_presenLoopScheduler.add(testHintRoutineEachFrame(snapshot));
    testHint_presenLoopScheduler.add(testHintRoutineEnd(snapshot));
    testHint_presenLoopScheduler.add(endLoopIteration(testHint_presenLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function testHint_presenLoopEnd() {
  psychoJS.experiment.removeLoop(testHint_presen);

  return Scheduler.Event.NEXT;
}


var test_loop;
function test_loopLoopBegin(test_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  test_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: testLoopCtrl, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: testLoopNum,
    seed: undefined, name: 'test_loop'
  });
  psychoJS.experiment.addLoop(test_loop); // add the loop to the experiment
  currentLoop = test_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  test_loop.forEach(function() {
    const snapshot = test_loop.getSnapshot();

    test_loopLoopScheduler.add(importConditions(snapshot));
    test_loopLoopScheduler.add(testTrialsRoutineBegin(snapshot));
    test_loopLoopScheduler.add(testTrialsRoutineEachFrame(snapshot));
    test_loopLoopScheduler.add(testTrialsRoutineEnd(snapshot));
    test_loopLoopScheduler.add(test_itiRoutineBegin(snapshot));
    test_loopLoopScheduler.add(test_itiRoutineEachFrame(snapshot));
    test_loopLoopScheduler.add(test_itiRoutineEnd(snapshot));
    test_loopLoopScheduler.add(endLoopIteration(test_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function test_loopLoopEnd() {
  psychoJS.experiment.removeLoop(test_loop);

  return Scheduler.Event.NEXT;
}


var expCtrl_loop;
function expCtrl_loopLoopBegin(expCtrl_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  expCtrl_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: endExpTag, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: undefined,
    seed: undefined, name: 'expCtrl_loop'
  });
  psychoJS.experiment.addLoop(expCtrl_loop); // add the loop to the experiment
  currentLoop = expCtrl_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  expCtrl_loop.forEach(function() {
    const snapshot = expCtrl_loop.getSnapshot();

    expCtrl_loopLoopScheduler.add(importConditions(snapshot));
    expCtrl_loopLoopScheduler.add(expENDRoutineBegin(snapshot));
    expCtrl_loopLoopScheduler.add(expENDRoutineEachFrame(snapshot));
    expCtrl_loopLoopScheduler.add(expENDRoutineEnd(snapshot));
    expCtrl_loopLoopScheduler.add(endLoopIteration(expCtrl_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function expCtrl_loopLoopEnd() {
  psychoJS.experiment.removeLoop(expCtrl_loop);

  return Scheduler.Event.NEXT;
}


function TBR_loopLoopEnd() {
  psychoJS.experiment.removeLoop(TBR_loop);

  return Scheduler.Event.NEXT;
}


var sch_loop;
function sch_loopLoopBegin(sch_loopLoopScheduler) {
  // set up handler to look after randomisation of conditions etc
  sch_loop = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'StimList/trialIndxList.xlsx',
    seed: undefined, name: 'sch_loop'
  });
  psychoJS.experiment.addLoop(sch_loop); // add the loop to the experiment
  currentLoop = sch_loop;  // we're now the current loop

  // Schedule all the trials in the trialList:
  sch_loop.forEach(function() {
    const snapshot = sch_loop.getSnapshot();

    sch_loopLoopScheduler.add(importConditions(snapshot));
    sch_loopLoopScheduler.add(searchTrialsRoutineBegin(snapshot));
    sch_loopLoopScheduler.add(searchTrialsRoutineEachFrame(snapshot));
    sch_loopLoopScheduler.add(searchTrialsRoutineEnd(snapshot));
    sch_loopLoopScheduler.add(endLoopIteration(sch_loopLoopScheduler, snapshot));
  });

  return Scheduler.Event.NEXT;
}


function sch_loopLoopEnd() {
  psychoJS.experiment.removeLoop(sch_loop);

  return Scheduler.Event.NEXT;
}


function blockOrdLoopEnd() {
  psychoJS.experiment.removeLoop(blockOrd);

  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var _instrResp1_allKeys;
var instr1Components;
function instr1RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instr1'-------
    t = 0;
    instr1Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrImg1.setImage(instrOrd1);
    instrResp1.keys = undefined;
    instrResp1.rt = undefined;
    _instrResp1_allKeys = [];
    // keep track of which components have finished
    instr1Components = [];
    instr1Components.push(instrImg1);
    instr1Components.push(instrResp1);
    
    instr1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


var continueRoutine;
function instr1RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instr1'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instr1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrImg1* updates
    if (t >= 0.0 && instrImg1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrImg1.tStart = t;  // (not accounting for frame time here)
      instrImg1.frameNStart = frameN;  // exact frame index
      
      instrImg1.setAutoDraw(true);
    }

    
    // *instrResp1* updates
    if (t >= 0.0 && instrResp1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrResp1.tStart = t;  // (not accounting for frame time here)
      instrResp1.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrResp1.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrResp1.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrResp1.clearEvents(); });
    }

    if (instrResp1.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrResp1.getKeys({keyList: ['space'], waitRelease: false});
      _instrResp1_allKeys = _instrResp1_allKeys.concat(theseKeys);
      if (_instrResp1_allKeys.length > 0) {
        instrResp1.keys = _instrResp1_allKeys[_instrResp1_allKeys.length - 1].name;  // just the last key pressed
        instrResp1.rt = _instrResp1_allKeys[_instrResp1_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instr1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instr1RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instr1'-------
    instr1Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrResp1.keys', instrResp1.keys);
    if (typeof instrResp1.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrResp1.rt', instrResp1.rt);
        routineTimer.reset();
        }
    
    instrResp1.stop();
    // the Routine "instr1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrResp5_allKeys;
var instrEG1Components;
function instrEG1RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrEG1'-------
    t = 0;
    instrEG1Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrResp5.keys = undefined;
    instrResp5.rt = undefined;
    _instrResp5_allKeys = [];
    // keep track of which components have finished
    instrEG1Components = [];
    instrEG1Components.push(instrImg5);
    instrEG1Components.push(instrResp5);
    
    instrEG1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEG1RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrEG1'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrEG1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrImg5* updates
    if (t >= 0.0 && instrImg5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrImg5.tStart = t;  // (not accounting for frame time here)
      instrImg5.frameNStart = frameN;  // exact frame index
      
      instrImg5.setAutoDraw(true);
    }

    
    // *instrResp5* updates
    if (t >= 0.0 && instrResp5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrResp5.tStart = t;  // (not accounting for frame time here)
      instrResp5.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrResp5.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrResp5.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrResp5.clearEvents(); });
    }

    if (instrResp5.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrResp5.getKeys({keyList: ['z'], waitRelease: false});
      _instrResp5_allKeys = _instrResp5_allKeys.concat(theseKeys);
      if (_instrResp5_allKeys.length > 0) {
        instrResp5.keys = _instrResp5_allKeys[_instrResp5_allKeys.length - 1].name;  // just the last key pressed
        instrResp5.rt = _instrResp5_allKeys[_instrResp5_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrEG1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEG1RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrEG1'-------
    instrEG1Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrResp5.keys', instrResp5.keys);
    if (typeof instrResp5.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrResp5.rt', instrResp5.rt);
        routineTimer.reset();
        }
    
    instrResp5.stop();
    // the Routine "instrEG1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrResp6_allKeys;
var instrEG2Components;
function instrEG2RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrEG2'-------
    t = 0;
    instrEG2Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrResp6.keys = undefined;
    instrResp6.rt = undefined;
    _instrResp6_allKeys = [];
    // keep track of which components have finished
    instrEG2Components = [];
    instrEG2Components.push(instrImg6);
    instrEG2Components.push(instrResp6);
    
    instrEG2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEG2RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrEG2'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrEG2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrImg6* updates
    if (t >= 0.0 && instrImg6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrImg6.tStart = t;  // (not accounting for frame time here)
      instrImg6.frameNStart = frameN;  // exact frame index
      
      instrImg6.setAutoDraw(true);
    }

    
    // *instrResp6* updates
    if (t >= 0.0 && instrResp6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrResp6.tStart = t;  // (not accounting for frame time here)
      instrResp6.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrResp6.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrResp6.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrResp6.clearEvents(); });
    }

    if (instrResp6.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrResp6.getKeys({keyList: ['m'], waitRelease: false});
      _instrResp6_allKeys = _instrResp6_allKeys.concat(theseKeys);
      if (_instrResp6_allKeys.length > 0) {
        instrResp6.keys = _instrResp6_allKeys[_instrResp6_allKeys.length - 1].name;  // just the last key pressed
        instrResp6.rt = _instrResp6_allKeys[_instrResp6_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrEG2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEG2RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrEG2'-------
    instrEG2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrResp6.keys', instrResp6.keys);
    if (typeof instrResp6.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrResp6.rt', instrResp6.rt);
        routineTimer.reset();
        }
    
    instrResp6.stop();
    // the Routine "instrEG2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrResp3_allKeys;
var instr2Components;
function instr2RoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instr2'-------
    t = 0;
    instr2Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrImg3.setImage(instrOrd2);
    instrResp3.keys = undefined;
    instrResp3.rt = undefined;
    _instrResp3_allKeys = [];
    // keep track of which components have finished
    instr2Components = [];
    instr2Components.push(instrImg3);
    instr2Components.push(instrResp3);
    
    instr2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instr2RoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instr2'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instr2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrImg3* updates
    if (t >= 0.0 && instrImg3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrImg3.tStart = t;  // (not accounting for frame time here)
      instrImg3.frameNStart = frameN;  // exact frame index
      
      instrImg3.setAutoDraw(true);
    }

    
    // *instrResp3* updates
    if (t >= 0.0 && instrResp3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrResp3.tStart = t;  // (not accounting for frame time here)
      instrResp3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrResp3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrResp3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrResp3.clearEvents(); });
    }

    if (instrResp3.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrResp3.getKeys({keyList: ['space'], waitRelease: false});
      _instrResp3_allKeys = _instrResp3_allKeys.concat(theseKeys);
      if (_instrResp3_allKeys.length > 0) {
        instrResp3.keys = _instrResp3_allKeys[_instrResp3_allKeys.length - 1].name;  // just the last key pressed
        instrResp3.rt = _instrResp3_allKeys[_instrResp3_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instr2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instr2RoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instr2'-------
    instr2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrResp3.keys', instrResp3.keys);
    if (typeof instrResp3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrResp3.rt', instrResp3.rt);
        routineTimer.reset();
        }
    
    instrResp3.stop();
    // the Routine "instr2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrEGresp_allKeys;
var instrSCHegComponents;
function instrSCHegRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrSCHeg'-------
    t = 0;
    instrSCHegClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrEGresp.keys = undefined;
    instrEGresp.rt = undefined;
    _instrEGresp_allKeys = [];
    // keep track of which components have finished
    instrSCHegComponents = [];
    instrSCHegComponents.push(instrEG);
    instrSCHegComponents.push(egITI);
    instrSCHegComponents.push(instrAnsHint);
    instrSCHegComponents.push(instrEGresp);
    
    instrSCHegComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


var frameRemains;
function instrSCHegRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrSCHeg'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrSCHegClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrEG* updates
    if (t >= 0.0 && instrEG.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrEG.tStart = t;  // (not accounting for frame time here)
      instrEG.frameNStart = frameN;  // exact frame index
      
      instrEG.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((instrEG.status === PsychoJS.Status.STARTED || instrEG.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      instrEG.setAutoDraw(false);
    }
    
    // *egITI* updates
    if (t >= 0.2 && egITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      egITI.tStart = t;  // (not accounting for frame time here)
      egITI.frameNStart = frameN;  // exact frame index
      
      egITI.setAutoDraw(true);
    }

    frameRemains = 0.2 + 0.3 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((egITI.status === PsychoJS.Status.STARTED || egITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      egITI.setAutoDraw(false);
    }
    
    // *instrAnsHint* updates
    if (t >= 0.5 && instrAnsHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrAnsHint.tStart = t;  // (not accounting for frame time here)
      instrAnsHint.frameNStart = frameN;  // exact frame index
      
      instrAnsHint.setAutoDraw(true);
    }

    
    // *instrEGresp* updates
    if (t >= 0.0 && instrEGresp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrEGresp.tStart = t;  // (not accounting for frame time here)
      instrEGresp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrEGresp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrEGresp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrEGresp.clearEvents(); });
    }

    if (instrEGresp.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrEGresp.getKeys({keyList: ['m'], waitRelease: false});
      _instrEGresp_allKeys = _instrEGresp_allKeys.concat(theseKeys);
      if (_instrEGresp_allKeys.length > 0) {
        instrEGresp.keys = _instrEGresp_allKeys[_instrEGresp_allKeys.length - 1].name;  // just the last key pressed
        instrEGresp.rt = _instrEGresp_allKeys[_instrEGresp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrSCHegComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrSCHegRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrSCHeg'-------
    instrSCHegComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrEGresp.keys', instrEGresp.keys);
    if (typeof instrEGresp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrEGresp.rt', instrEGresp.rt);
        routineTimer.reset();
        }
    
    instrEGresp.stop();
    // the Routine "instrSCHeg" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrHint2Resp_allKeys;
var instrHintEndComponents;
function instrHintEndRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrHintEnd'-------
    t = 0;
    instrHintEndClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrHint2Resp.keys = undefined;
    instrHint2Resp.rt = undefined;
    _instrHint2Resp_allKeys = [];
    // keep track of which components have finished
    instrHintEndComponents = [];
    instrHintEndComponents.push(instrHint2);
    instrHintEndComponents.push(instrHint2Resp);
    
    instrHintEndComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrHintEndRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrHintEnd'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrHintEndClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrHint2* updates
    if (t >= 0.0 && instrHint2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrHint2.tStart = t;  // (not accounting for frame time here)
      instrHint2.frameNStart = frameN;  // exact frame index
      
      instrHint2.setAutoDraw(true);
    }

    
    // *instrHint2Resp* updates
    if (t >= 0.0 && instrHint2Resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrHint2Resp.tStart = t;  // (not accounting for frame time here)
      instrHint2Resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrHint2Resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrHint2Resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrHint2Resp.clearEvents(); });
    }

    if (instrHint2Resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrHint2Resp.getKeys({keyList: ['space'], waitRelease: false});
      _instrHint2Resp_allKeys = _instrHint2Resp_allKeys.concat(theseKeys);
      if (_instrHint2Resp_allKeys.length > 0) {
        instrHint2Resp.keys = _instrHint2Resp_allKeys[_instrHint2Resp_allKeys.length - 1].name;  // just the last key pressed
        instrHint2Resp.rt = _instrHint2Resp_allKeys[_instrHint2Resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrHintEndComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrHintEndRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrHintEnd'-------
    instrHintEndComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrHint2Resp.keys', instrHint2Resp.keys);
    if (typeof instrHint2Resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrHint2Resp.rt', instrHint2Resp.rt);
        routineTimer.reset();
        }
    
    instrHint2Resp.stop();
    // the Routine "instrHintEnd" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var instrInterComponents;
function instrInterRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrInter'-------
    t = 0;
    instrInterClock.reset(); // clock
    frameN = -1;
    routineTimer.add(1.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    instrInterComponents = [];
    instrInterComponents.push(imageITI);
    
    instrInterComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrInterRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrInter'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrInterClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *imageITI* updates
    if (t >= 0.0 && imageITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      imageITI.tStart = t;  // (not accounting for frame time here)
      imageITI.frameNStart = frameN;  // exact frame index
      
      imageITI.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((imageITI.status === PsychoJS.Status.STARTED || imageITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      imageITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrInterComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrInterRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrInter'-------
    instrInterComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var _instrRespSCH_allKeys;
var instrSCHComponents;
function instrSCHRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrSCH'-------
    t = 0;
    instrSCHClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    instrImgSCH.setImage(instrOrd3);
    instrRespSCH.keys = undefined;
    instrRespSCH.rt = undefined;
    _instrRespSCH_allKeys = [];
    // keep track of which components have finished
    instrSCHComponents = [];
    instrSCHComponents.push(instrImgSCH);
    instrSCHComponents.push(instrITI);
    instrSCHComponents.push(instrRespSCH);
    
    instrSCHComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrSCHRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrSCH'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrSCHClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrImgSCH* updates
    if (t >= 0.0 && instrImgSCH.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrImgSCH.tStart = t;  // (not accounting for frame time here)
      instrImgSCH.frameNStart = frameN;  // exact frame index
      
      instrImgSCH.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((instrImgSCH.status === PsychoJS.Status.STARTED || instrImgSCH.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      instrImgSCH.setAutoDraw(false);
    }
    
    // *instrITI* updates
    if (t >= 0.2 && instrITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrITI.tStart = t;  // (not accounting for frame time here)
      instrITI.frameNStart = frameN;  // exact frame index
      
      instrITI.setAutoDraw(true);
    }

    frameRemains = 0.2 + 1.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((instrITI.status === PsychoJS.Status.STARTED || instrITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      instrITI.setAutoDraw(false);
    }
    
    // *instrRespSCH* updates
    if (t >= 0 && instrRespSCH.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrRespSCH.tStart = t;  // (not accounting for frame time here)
      instrRespSCH.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrRespSCH.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrRespSCH.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrRespSCH.clearEvents(); });
    }

    frameRemains = 0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((instrRespSCH.status === PsychoJS.Status.STARTED || instrRespSCH.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      instrRespSCH.status = PsychoJS.Status.FINISHED;
  }

    if (instrRespSCH.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrRespSCH.getKeys({keyList: ['z', 'm'], waitRelease: false});
      _instrRespSCH_allKeys = _instrRespSCH_allKeys.concat(theseKeys);
      if (_instrRespSCH_allKeys.length > 0) {
        instrRespSCH.keys = _instrRespSCH_allKeys[_instrRespSCH_allKeys.length - 1].name;  // just the last key pressed
        instrRespSCH.rt = _instrRespSCH_allKeys[_instrRespSCH_allKeys.length - 1].rt;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrSCHComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrSCHRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrSCH'-------
    instrSCHComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrRespSCH.keys', instrRespSCH.keys);
    if (typeof instrRespSCH.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrRespSCH.rt', instrRespSCH.rt);
        }
    
    instrRespSCH.stop();
    return Scheduler.Event.NEXT;
  };
}


var _instrHint3Resp_allKeys;
var instrHintExpComponents;
function instrHintExpRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrHintExp'-------
    t = 0;
    instrHintExpClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrHint3Resp.keys = undefined;
    instrHint3Resp.rt = undefined;
    _instrHint3Resp_allKeys = [];
    // keep track of which components have finished
    instrHintExpComponents = [];
    instrHintExpComponents.push(instrHint3);
    instrHintExpComponents.push(instrHint3Resp);
    
    instrHintExpComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrHintExpRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrHintExp'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrHintExpClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instrHint3* updates
    if (t >= 0.0 && instrHint3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrHint3.tStart = t;  // (not accounting for frame time here)
      instrHint3.frameNStart = frameN;  // exact frame index
      
      instrHint3.setAutoDraw(true);
    }

    
    // *instrHint3Resp* updates
    if (t >= 0.0 && instrHint3Resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrHint3Resp.tStart = t;  // (not accounting for frame time here)
      instrHint3Resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrHint3Resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrHint3Resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrHint3Resp.clearEvents(); });
    }

    if (instrHint3Resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrHint3Resp.getKeys({keyList: ['space'], waitRelease: false});
      _instrHint3Resp_allKeys = _instrHint3Resp_allKeys.concat(theseKeys);
      if (_instrHint3Resp_allKeys.length > 0) {
        instrHint3Resp.keys = _instrHint3Resp_allKeys[_instrHint3Resp_allKeys.length - 1].name;  // just the last key pressed
        instrHint3Resp.rt = _instrHint3Resp_allKeys[_instrHint3Resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrHintExpComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrHintExpRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrHintExp'-------
    instrHintExpComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrHint3Resp.keys', instrHint3Resp.keys);
    if (typeof instrHint3Resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrHint3Resp.rt', instrHint3Resp.rt);
        routineTimer.reset();
        }
    
    instrHint3Resp.stop();
    // the Routine "instrHintExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _instrResp10_allKeys;
var instrEndComponents;
function instrEndRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'instrEnd'-------
    t = 0;
    instrEndClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    instrResp10.keys = undefined;
    instrResp10.rt = undefined;
    _instrResp10_allKeys = [];
    // keep track of which components have finished
    instrEndComponents = [];
    instrEndComponents.push(instr10);
    instrEndComponents.push(instrResp10);
    
    instrEndComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEndRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'instrEnd'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = instrEndClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instr10* updates
    if (t >= 0.0 && instr10.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instr10.tStart = t;  // (not accounting for frame time here)
      instr10.frameNStart = frameN;  // exact frame index
      
      instr10.setAutoDraw(true);
    }

    
    // *instrResp10* updates
    if (t >= 0.0 && instrResp10.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instrResp10.tStart = t;  // (not accounting for frame time here)
      instrResp10.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { instrResp10.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { instrResp10.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { instrResp10.clearEvents(); });
    }

    if (instrResp10.status === PsychoJS.Status.STARTED) {
      let theseKeys = instrResp10.getKeys({keyList: ['space'], waitRelease: false});
      _instrResp10_allKeys = _instrResp10_allKeys.concat(theseKeys);
      if (_instrResp10_allKeys.length > 0) {
        instrResp10.keys = _instrResp10_allKeys[_instrResp10_allKeys.length - 1].name;  // just the last key pressed
        instrResp10.rt = _instrResp10_allKeys[_instrResp10_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instrEndComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instrEndRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'instrEnd'-------
    instrEndComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instrResp10.keys', instrResp10.keys);
    if (typeof instrResp10.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('instrResp10.rt', instrResp10.rt);
        routineTimer.reset();
        }
    
    instrResp10.stop();
    // the Routine "instrEnd" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var pracStartComponents;
function pracStartRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pracStart'-------
    t = 0;
    pracStartClock.reset(); // clock
    frameN = -1;
    routineTimer.add(1.700000);
    // update component parameters for each repeat
    // keep track of which components have finished
    pracStartComponents = [];
    pracStartComponents.push(pStartHint);
    pracStartComponents.push(pITI);
    
    pracStartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pracStartRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pracStart'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pracStartClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pStartHint* updates
    if (t >= 0.0 && pStartHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pStartHint.tStart = t;  // (not accounting for frame time here)
      pStartHint.frameNStart = frameN;  // exact frame index
      
      pStartHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pStartHint.status === PsychoJS.Status.STARTED || pStartHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pStartHint.setAutoDraw(false);
    }
    
    // *pITI* updates
    if (t >= 1.2 && pITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pITI.tStart = t;  // (not accounting for frame time here)
      pITI.frameNStart = frameN;  // exact frame index
      
      pITI.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pITI.status === PsychoJS.Status.STARTED || pITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pracStartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pracStartRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pracStart'-------
    pracStartComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var pmemoText;
var ptestText;
var pstLoopCtrl;
var pthLoopCtrl;
var ptestLoopCtrl;
var pTBRcount;
var pACClist;
var pTBRctrlComponents;
function pTBRctrlRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pTBRctrl'-------
    t = 0;
    pTBRctrlClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    pmemoText = "Memorizing";
    ptestText = "Testing 1";
    pstLoopCtrl = 1;
    pthLoopCtrl = 1;
    ptestLoopCtrl = 1;
    pTBRcount = 0;
    pACClist = {};
    
    // keep track of which components have finished
    pTBRctrlComponents = [];
    
    pTBRctrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pTBRctrlRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pTBRctrl'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pTBRctrlClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pTBRctrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pTBRctrlRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pTBRctrl'-------
    pTBRctrlComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "pTBRctrl" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var pTestFile;
var pACCsum;
var ptN;
var jdgTestFileComponents;
function jdgTestFileRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'jdgTestFile'-------
    t = 0;
    jdgTestFileClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    if ((pTBRcount === 0)) {
        pTestFile = pTest1;
    } else {
        if ((pTBRcount === 1)) {
            pTestFile = pTest2;
        } else {
            if ((pTBRcount === 3)) {
                pTestFile = pTest3;
            }
        }
    }
    pACCsum = 0;
    ptN = 0;
    
    // keep track of which components have finished
    jdgTestFileComponents = [];
    
    jdgTestFileComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function jdgTestFileRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'jdgTestFile'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = jdgTestFileClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    jdgTestFileComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function jdgTestFileRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'jdgTestFile'-------
    jdgTestFileComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "jdgTestFile" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var pStudyHintComponents;
function pStudyHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pStudyHint'-------
    t = 0;
    pStudyHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    psHint.setText(pmemoText);
    // keep track of which components have finished
    pStudyHintComponents = [];
    pStudyHintComponents.push(psHint);
    pStudyHintComponents.push(psFix);
    
    pStudyHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pStudyHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pStudyHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pStudyHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *psHint* updates
    if (t >= 0.0 && psHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      psHint.tStart = t;  // (not accounting for frame time here)
      psHint.frameNStart = frameN;  // exact frame index
      
      psHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((psHint.status === PsychoJS.Status.STARTED || psHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      psHint.setAutoDraw(false);
    }
    
    // *psFix* updates
    if (t >= 1.2 && psFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      psFix.tStart = t;  // (not accounting for frame time here)
      psFix.frameNStart = frameN;  // exact frame index
      
      psFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((psFix.status === PsychoJS.Status.STARTED || psFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      psFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pStudyHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pStudyHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pStudyHint'-------
    pStudyHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var pStudyTrialsComponents;
function pStudyTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pStudyTrials'-------
    t = 0;
    pStudyTrialsClock.reset(); // clock
    frameN = -1;
    routineTimer.add(3.950000);
    // update component parameters for each repeat
    pStudyImg.setImage(pTarg);
    // keep track of which components have finished
    pStudyTrialsComponents = [];
    pStudyTrialsComponents.push(pStudyImg);
    pStudyTrialsComponents.push(pStdITI);
    
    pStudyTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pStudyTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pStudyTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pStudyTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pStudyImg* updates
    if (t >= 0.0 && pStudyImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pStudyImg.tStart = t;  // (not accounting for frame time here)
      pStudyImg.frameNStart = frameN;  // exact frame index
      
      pStudyImg.setAutoDraw(true);
    }

    frameRemains = 0.0 + 3 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pStudyImg.status === PsychoJS.Status.STARTED || pStudyImg.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pStudyImg.setAutoDraw(false);
    }
    
    // *pStdITI* updates
    if (t >= 3 && pStdITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pStdITI.tStart = t;  // (not accounting for frame time here)
      pStdITI.frameNStart = frameN;  // exact frame index
      
      pStdITI.setAutoDraw(true);
    }

    frameRemains = 3 + 0.95 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pStdITI.status === PsychoJS.Status.STARTED || pStdITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pStdITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pStudyTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pStudyTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pStudyTrials'-------
    pStudyTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var pTestHintComponents;
function pTestHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pTestHint'-------
    t = 0;
    pTestHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    ptHint.setText(ptestText);
    // keep track of which components have finished
    pTestHintComponents = [];
    pTestHintComponents.push(ptHint);
    pTestHintComponents.push(ptFix);
    
    pTestHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pTestHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pTestHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pTestHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *ptHint* updates
    if (t >= 0.0 && ptHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ptHint.tStart = t;  // (not accounting for frame time here)
      ptHint.frameNStart = frameN;  // exact frame index
      
      ptHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ptHint.status === PsychoJS.Status.STARTED || ptHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ptHint.setAutoDraw(false);
    }
    
    // *ptFix* updates
    if (t >= 1.2 && ptFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ptFix.tStart = t;  // (not accounting for frame time here)
      ptFix.frameNStart = frameN;  // exact frame index
      
      ptFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ptFix.status === PsychoJS.Status.STARTED || ptFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ptFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pTestHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pTestHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pTestHint'-------
    pTestHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var _pTestResp_allKeys;
var pTestTrialsComponents;
function pTestTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pTestTrials'-------
    t = 0;
    pTestTrialsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    pTestImg.setImage(ptStim);
    pTestResp.keys = undefined;
    pTestResp.rt = undefined;
    _pTestResp_allKeys = [];
    // keep track of which components have finished
    pTestTrialsComponents = [];
    pTestTrialsComponents.push(pTestImg);
    pTestTrialsComponents.push(pTestResp);
    
    pTestTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pTestTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pTestTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pTestTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pTestImg* updates
    if (t >= 0.0 && pTestImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pTestImg.tStart = t;  // (not accounting for frame time here)
      pTestImg.frameNStart = frameN;  // exact frame index
      
      pTestImg.setAutoDraw(true);
    }

    
    // *pTestResp* updates
    if (t >= 0.0 && pTestResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pTestResp.tStart = t;  // (not accounting for frame time here)
      pTestResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { pTestResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { pTestResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { pTestResp.clearEvents(); });
    }

    if (pTestResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = pTestResp.getKeys({keyList: ['z', 'm'], waitRelease: false});
      _pTestResp_allKeys = _pTestResp_allKeys.concat(theseKeys);
      if (_pTestResp_allKeys.length > 0) {
        pTestResp.keys = _pTestResp_allKeys[0].name;  // just the first key pressed
        pTestResp.rt = _pTestResp_allKeys[0].rt;
        // was this correct?
        if (pTestResp.keys == pCorAns) {
            pTestResp.corr = 1;
        } else {
            pTestResp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pTestTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pTestTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pTestTrials'-------
    pTestTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // was no response the correct answer?!
    if (pTestResp.keys === undefined) {
      if (['None','none',undefined].includes(pCorAns)) {
         pTestResp.corr = 1;  // correct non-response
      } else {
         pTestResp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for thisExp (ExperimentHandler)
    psychoJS.experiment.addData('pTestResp.keys', pTestResp.keys);
    psychoJS.experiment.addData('pTestResp.corr', pTestResp.corr);
    if (typeof pTestResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('pTestResp.rt', pTestResp.rt);
        routineTimer.reset();
        }
    
    pTestResp.stop();
    // the Routine "pTestTrials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var pfbkText;
var pclr;
var pTextITIComponents;
function pTextITIRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pTextITI'-------
    t = 0;
    pTextITIClock.reset(); // clock
    frameN = -1;
    routineTimer.add(1.100000);
    // update component parameters for each repeat
    if ((pTestResp.corr === 1)) {
        pfbkText = "Correct!";
        pclr = "green";
    } else {
        pfbkText = "Wrong!";
        pclr = "red";
    }
    
    pfbkTest.setColor(new util.Color(pclr));
    pfbkTest.setText(pfbkText);
    // keep track of which components have finished
    pTextITIComponents = [];
    pTextITIComponents.push(pfbkTest);
    pTextITIComponents.push(pTestITI);
    
    pTextITIComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pTextITIRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pTextITI'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pTextITIClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pfbkTest* updates
    if (t >= 0.0 && pfbkTest.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pfbkTest.tStart = t;  // (not accounting for frame time here)
      pfbkTest.frameNStart = frameN;  // exact frame index
      
      pfbkTest.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.6 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pfbkTest.status === PsychoJS.Status.STARTED || pfbkTest.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pfbkTest.setAutoDraw(false);
    }
    
    // *pTestITI* updates
    if (t >= 0.6 && pTestITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pTestITI.tStart = t;  // (not accounting for frame time here)
      pTestITI.frameNStart = frameN;  // exact frame index
      
      pTestITI.setAutoDraw(true);
    }

    frameRemains = 0.6 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pTestITI.status === PsychoJS.Status.STARTED || pTestITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pTestITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pTextITIComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pTextITIRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pTextITI'-------
    pTextITIComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    pACCsum = (pACCsum + pTestResp.corr);
    ptN = (ptN + 1);
    
    return Scheduler.Event.NEXT;
  };
}


var pendExpTag;
var pACCcountComponents;
function pACCcountRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pACCcount'-------
    t = 0;
    pACCcountClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    if ((ptN === 0)) {
        ptN = 1;
    }
    pACClist[pTBRcount.toString()] = (pACCsum / ptN);
    pTBRcount = (pTBRcount + 1);
    if ((pTBRcount === 1)) {
        if ((pACClist["0"] >= 0.8)) {
            pendExpTag = 0;
            pstLoopCtrl = 0;
            pthLoopCtrl = 1;
            ptestLoopCtrl = 1;
            ptestText = "Testing 2";
        } else {
            if ((pACClist["0"] < 0.8)) {
                pendExpTag = 0;
                pstLoopCtrl = 1;
                pthLoopCtrl = 1;
                ptestLoopCtrl = 1;
                pmemoText = "Re-Memorizing";
                ptestText = "Testing 1";
            }
        }
    } else {
        if ((((pTBRcount >= 2) && (pACClist["0"] >= 0.8)) && (pACClist["1"] >= 0.8))) {
            pendExpTag = 0;
            pstLoopCtrl = 0;
            pthLoopCtrl = 0;
            ptestLoopCtrl = 0;
        } else {
            if ((pTBRcount === 2)) {
                if (((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8))) {
                    pendExpTag = 0;
                    pstLoopCtrl = 1;
                    pthLoopCtrl = 1;
                    ptestLoopCtrl = 1;
                    pmemoText = "Re-Memorizing";
                    ptestText = "Testing 2";
                } else {
                    if (((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8))) {
                        pendExpTag = 0;
                        pstLoopCtrl = 0;
                        pthLoopCtrl = 1;
                        ptestLoopCtrl = 1;
                        ptestText = "Testing 2";
                    } else {
                        if (((pACClist["0"] < 0.8) && (pACClist["1"] < 0.8))) {
                            pendExpTag = 0;
                            pstLoopCtrl = 1;
                            pthLoopCtrl = 1;
                            ptestLoopCtrl = 1;
                            pmemoText = "Re-Memorizing";
                            ptestText = "Testing 1";
                        }
                    }
                }
            } else {
                if ((pTBRcount === 3)) {
                    if (((((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] < 0.8)) || (((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] < 0.8)))) {
                        pendExpTag = 0;
                        pstLoopCtrl = 1;
                        pthLoopCtrl = 1;
                        ptestLoopCtrl = 1;
                        ptestText = "Testing 2";
                    } else {
                        if ((((pACClist["0"] < 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8))) {
                            pendExpTag = 0;
                            pstLoopCtrl = 0;
                            pthLoopCtrl = 1;
                            ptestLoopCtrl = 1;
                            ptestText = "Testing 2";
                        } else {
                            if (((((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] >= 0.8)) || (((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8)))) {
                                pendExpTag = 0;
                                pstLoopCtrl = 0;
                                pthLoopCtrl = 1;
                                ptestLoopCtrl = 1;
                                ptestText = "Testing 3";
                            } else {
                                if ((((pACClist["0"] < 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] < 0.8))) {
                                    pendExpTag = 1;
                                    pstLoopCtrl = 0;
                                    pthLoopCtrl = 0;
                                    ptestLoopCtrl = 0;
                                }
                            }
                        }
                    }
                } else {
                    if (((pTBRcount >= 4) && (((((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] >= 0.8)) || ((((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] >= 0.8))))) {
                        pendExpTag = 0;
                        pstLoopCtrl = 0;
                        pthLoopCtrl = 0;
                        ptestLoopCtrl = 0;
                    } else {
                        if ((pTBRcount === 4)) {
                            if (((((((pACClist["0"] < 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] >= 0.8)) || ((((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] < 0.8)) && (pACClist["3"] >= 0.8))) || ((((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] < 0.8)) && (pACClist["3"] >= 0.8)))) {
                                pendExpTag = 0;
                                pstLoopCtrl = 0;
                                pthLoopCtrl = 1;
                                ptestLoopCtrl = 1;
                                ptestText = "Testing 3";
                            } else {
                                if ((((((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] < 0.8)) || ((((pACClist["0"] > 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] < 0.8)))) {
                                    pendExpTag = 0;
                                    pstLoopCtrl = 1;
                                    pthLoopCtrl = 1;
                                    ptestLoopCtrl = 1;
                                    ptestText = "Testing 3";
                                } else {
                                    if (((((((pACClist["0"] >= 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] < 0.8)) && (pACClist["3"] < 0.8)) || ((((pACClist["0"] < 0.8) && (pACClist["1"] >= 0.8)) && (pACClist["2"] < 0.8)) && (pACClist["3"] < 0.8))) || ((((pACClist["0"] < 0.8) && (pACClist["1"] < 0.8)) && (pACClist["2"] >= 0.8)) && (pACClist["3"] < 0.8)))) {
                                        pendExpTag = 1;
                                        pstLoopCtrl = 0;
                                        pthLoopCtrl = 0;
                                        ptestLoopCtrl = 0;
                                    }
                                }
                            }
                        } else {
                            if (((pTBRcount === 5) && (pACClist["4"] < 0.8))) {
                                pendExpTag = 0;
                                pstLoopCtrl = 0;
                                pthLoopCtrl = 0;
                                ptestLoopCtrl = 0;
                            } else {
                                if (((pTBRcount === 5) && (pACClist["4"] >= 0.8))) {
                                    pendExpTag = 0;
                                    pstLoopCtrl = 0;
                                    pthLoopCtrl = 0;
                                    ptestLoopCtrl = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // keep track of which components have finished
    pACCcountComponents = [];
    
    pACCcountComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pACCcountRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pACCcount'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pACCcountClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pACCcountComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pACCcountRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pACCcount'-------
    pACCcountComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "pACCcount" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _endConResp_allKeys;
var endHintComponents;
function endHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'endHint'-------
    t = 0;
    endHintClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    endConResp.keys = undefined;
    endConResp.rt = undefined;
    _endConResp_allKeys = [];
    // keep track of which components have finished
    endHintComponents = [];
    endHintComponents.push(endHintImg);
    endHintComponents.push(endConResp);
    
    endHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function endHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'endHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = endHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *endHintImg* updates
    if (t >= 0.0 && endHintImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      endHintImg.tStart = t;  // (not accounting for frame time here)
      endHintImg.frameNStart = frameN;  // exact frame index
      
      endHintImg.setAutoDraw(true);
    }

    
    // *endConResp* updates
    if (t >= 0.0 && endConResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      endConResp.tStart = t;  // (not accounting for frame time here)
      endConResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { endConResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { endConResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { endConResp.clearEvents(); });
    }

    if (endConResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = endConResp.getKeys({keyList: ['space'], waitRelease: false});
      _endConResp_allKeys = _endConResp_allKeys.concat(theseKeys);
      if (_endConResp_allKeys.length > 0) {
        endConResp.keys = _endConResp_allKeys[_endConResp_allKeys.length - 1].name;  // just the last key pressed
        endConResp.rt = _endConResp_allKeys[_endConResp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    endHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var endHintTag;
function endHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'endHint'-------
    endHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('endConResp.keys', endConResp.keys);
    if (typeof endConResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('endConResp.rt', endConResp.rt);
        routineTimer.reset();
        }
    
    endConResp.stop();
    endHintTag = 0;
    
    // the Routine "endHint" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var pACCschSum;
var pACCschSumR;
var pSearchHintComponents;
function pSearchHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pSearchHint'-------
    t = 0;
    pSearchHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    pACCschSum = 0;
    pACCschSumR = 0;
    
    // keep track of which components have finished
    pSearchHintComponents = [];
    pSearchHintComponents.push(sHint);
    pSearchHintComponents.push(sFix);
    
    pSearchHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pSearchHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pSearchHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pSearchHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *sHint* updates
    if (t >= 0.0 && sHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sHint.tStart = t;  // (not accounting for frame time here)
      sHint.frameNStart = frameN;  // exact frame index
      
      sHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((sHint.status === PsychoJS.Status.STARTED || sHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      sHint.setAutoDraw(false);
    }
    
    // *sFix* updates
    if (t >= 1.2 && sFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sFix.tStart = t;  // (not accounting for frame time here)
      sFix.frameNStart = frameN;  // exact frame index
      
      sFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((sFix.status === PsychoJS.Status.STARTED || sFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      sFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pSearchHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pSearchHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pSearchHint'-------
    pSearchHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var _psResp_allKeys;
var pSearchTrialsComponents;
function pSearchTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pSearchTrials'-------
    t = 0;
    pSearchTrialsClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    psImg.setImage(pStim);
    psResp.keys = undefined;
    psResp.rt = undefined;
    _psResp_allKeys = [];
    // keep track of which components have finished
    pSearchTrialsComponents = [];
    pSearchTrialsComponents.push(psImg);
    pSearchTrialsComponents.push(psITI);
    pSearchTrialsComponents.push(psResp);
    
    pSearchTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pSearchTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pSearchTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pSearchTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *psImg* updates
    if (t >= 0.0 && psImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      psImg.tStart = t;  // (not accounting for frame time here)
      psImg.frameNStart = frameN;  // exact frame index
      
      psImg.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((psImg.status === PsychoJS.Status.STARTED || psImg.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      psImg.setAutoDraw(false);
    }
    
    // *psITI* updates
    if (t >= 0.2 && psITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      psITI.tStart = t;  // (not accounting for frame time here)
      psITI.frameNStart = frameN;  // exact frame index
      
      psITI.setAutoDraw(true);
    }

    frameRemains = 0.2 + 1.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((psITI.status === PsychoJS.Status.STARTED || psITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      psITI.setAutoDraw(false);
    }
    
    // *psResp* updates
    if (t >= 0.0 && psResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      psResp.tStart = t;  // (not accounting for frame time here)
      psResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { psResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { psResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { psResp.clearEvents(); });
    }

    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((psResp.status === PsychoJS.Status.STARTED || psResp.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      psResp.status = PsychoJS.Status.FINISHED;
  }

    if (psResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = psResp.getKeys({keyList: ['z', 'm'], waitRelease: false});
      _psResp_allKeys = _psResp_allKeys.concat(theseKeys);
      if (_psResp_allKeys.length > 0) {
        psResp.keys = _psResp_allKeys[0].name;  // just the first key pressed
        psResp.rt = _psResp_allKeys[0].rt;
        // was this correct?
        if (psResp.keys == pCorAns) {
            psResp.corr = 1;
        } else {
            psResp.corr = 0;
        }
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pSearchTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pSearchTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pSearchTrials'-------
    pSearchTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // was no response the correct answer?!
    if (psResp.keys === undefined) {
      if (['None','none',undefined].includes(pCorAns)) {
         psResp.corr = 1;  // correct non-response
      } else {
         psResp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for thisExp (ExperimentHandler)
    psychoJS.experiment.addData('psResp.keys', psResp.keys);
    psychoJS.experiment.addData('psResp.corr', psResp.corr);
    if (typeof psResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('psResp.rt', psResp.rt);
        }
    
    psResp.stop();
    if (((psResp.corr === 1) && (psResp.keys === "z"))) {
        pACCschSum = (pACCschSum + 1);
    } else {
        if (((psResp.corr === 1) && (psResp.keys === "m"))) {
            pACCschSumR = (pACCschSumR + 1);
        }
    }
    
    return Scheduler.Event.NEXT;
  };
}


var pACCsch;
var pACCschStr;
var pACCstr;
var pACCtext1;
var pACCschR;
var pACCschStrR;
var pACCstrR;
var pACCtext2;
var pACCpresenComponents;
function pACCpresenRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'pACCpresen'-------
    t = 0;
    pACCpresenClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    pACCsch = ((pACCschSum / 4) * 100);
    pACCschStr = pACCsch.toString();
    pACCstr = "";
    if ((pACCschStr.length > 5)) {
        for (var paN = 0, _pj_a = 5; (paN < _pj_a); paN += 1) {
            pACCstr = (pACCstr + pACCschStr[paN]);
        }
    } else {
        for (var paN = 0, _pj_a = pACCschStr.length; (paN < _pj_a); paN += 1) {
            pACCstr = (pACCstr + pACCschStr[paN]);
        }
    }
    pACCtext1 = (("You detected " + pACCstr) + "% of the targets");
    pACCschR = ((pACCschSumR / 16) * 100);
    pACCschStrR = pACCschR.toString();
    pACCstrR = "";
    if ((pACCschStrR.length > 5)) {
        for (var paNR = 0, _pj_a = 5; (paNR < _pj_a); paNR += 1) {
            pACCstrR = (pACCstrR + pACCschStrR[paNR]);
        }
    } else {
        for (var paNR = 0, _pj_a = pACCschStrR.length; (paNR < _pj_a); paNR += 1) {
            pACCstrR = (pACCstrR + pACCschStrR[paNR]);
        }
    }
    pACCtext2 = (("You rejected " + pACCstrR) + "% of the distractors");
    
    pACCpresen1.setText(pACCtext1);
    pACCpresen2.setText(pACCtext2);
    // keep track of which components have finished
    pACCpresenComponents = [];
    pACCpresenComponents.push(pACCpresen1);
    pACCpresenComponents.push(pACCpresen2);
    
    pACCpresenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function pACCpresenRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'pACCpresen'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pACCpresenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pACCpresen1* updates
    if (t >= 0.0 && pACCpresen1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pACCpresen1.tStart = t;  // (not accounting for frame time here)
      pACCpresen1.frameNStart = frameN;  // exact frame index
      
      pACCpresen1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pACCpresen1.status === PsychoJS.Status.STARTED || pACCpresen1.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pACCpresen1.setAutoDraw(false);
    }
    
    // *pACCpresen2* updates
    if (t >= 0.0 && pACCpresen2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pACCpresen2.tStart = t;  // (not accounting for frame time here)
      pACCpresen2.frameNStart = frameN;  // exact frame index
      
      pACCpresen2.setAutoDraw(true);
    }

    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((pACCpresen2.status === PsychoJS.Status.STARTED || pACCpresen2.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      pACCpresen2.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pACCpresenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pACCpresenRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'pACCpresen'-------
    pACCpresenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var _expStaResp_allKeys;
var expStartComponents;
function expStartRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'expStart'-------
    t = 0;
    expStartClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    expStaResp.keys = undefined;
    expStaResp.rt = undefined;
    _expStaResp_allKeys = [];
    // keep track of which components have finished
    expStartComponents = [];
    expStartComponents.push(formalExp);
    expStartComponents.push(expStaResp);
    
    expStartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function expStartRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'expStart'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = expStartClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *formalExp* updates
    if (t >= 0.0 && formalExp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      formalExp.tStart = t;  // (not accounting for frame time here)
      formalExp.frameNStart = frameN;  // exact frame index
      
      formalExp.setAutoDraw(true);
    }

    
    // *expStaResp* updates
    if (t >= 0.0 && expStaResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      expStaResp.tStart = t;  // (not accounting for frame time here)
      expStaResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { expStaResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { expStaResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { expStaResp.clearEvents(); });
    }

    if (expStaResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = expStaResp.getKeys({keyList: ['space'], waitRelease: false});
      _expStaResp_allKeys = _expStaResp_allKeys.concat(theseKeys);
      if (_expStaResp_allKeys.length > 0) {
        expStaResp.keys = _expStaResp_allKeys[_expStaResp_allKeys.length - 1].name;  // just the last key pressed
        expStaResp.rt = _expStaResp_allKeys[_expStaResp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    expStartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function expStartRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'expStart'-------
    expStartComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('expStaResp.keys', expStaResp.keys);
    if (typeof expStaResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('expStaResp.rt', expStaResp.rt);
        routineTimer.reset();
        }
    
    expStaResp.stop();
    // the Routine "expStart" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var distrN;
var targetN;
var trialN;
var oldList;
var preExpComponents;
function preExpRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'preExp'-------
    t = 0;
    preExpClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    distrN = 24;
    targetN = 12;
    trialN = 60;
    oldList = [];
    
    // keep track of which components have finished
    preExpComponents = [];
    
    preExpComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function preExpRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'preExp'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = preExpClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    preExpComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function preExpRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'preExp'-------
    preExpComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "preExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var cateTag;
var altCateTag;
var oldTargSubList;
var oldSubList;
var targetList;
var targetDict;
var targDistrDictAll;
var altDistrDictAll;
var test1Dict;
var test2Dict;
var test3Dict;
var searchDict;
var searchCate;
var searchSubCate;
var subCateT;
var subCateTD;
var subCateD;
var subCount;
var stimDcount;
var targKey;
var tdAllCount;
var dAllCount;
var test1Key;
var test2Key;
var test3Key;
var subCountBeginComponents;
function subCountBeginRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'subCountBegin'-------
    t = 0;
    subCountBeginClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    cateTag = stimCate;
    if ((cateTag === "Animals")) {
        altCateTag = "Objects";
    } else {
        altCateTag = "Animals";
    }
    oldTargSubList = [];
    oldSubList = [];
    targetList = [];
    targetDict = {};
    targDistrDictAll = {};
    altDistrDictAll = {};
    test1Dict = {};
    test2Dict = {};
    test3Dict = {};
    searchDict = {};
    searchCate = {};
    searchSubCate = {};
    subCateT = {};
    subCateTD = {};
    subCateD = {};
    subCount = 0;
    stimDcount = 0;
    targKey = 0;
    tdAllCount = 0;
    dAllCount = 0;
    test1Key = setsize;
    test2Key = setsize;
    test3Key = setsize;
    
    // keep track of which components have finished
    subCountBeginComponents = [];
    
    subCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function subCountBeginRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'subCountBegin'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = subCountBeginClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    subCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function subCountBeginRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'subCountBegin'-------
    subCountBeginComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "subCountBegin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var stimCount;
var stimCountBeginComponents;
function stimCountBeginRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'stimCountBegin'-------
    t = 0;
    stimCountBeginClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    stimCount = 0;
    
    // keep track of which components have finished
    stimCountBeginComponents = [];
    
    stimCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function stimCountBeginRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'stimCountBegin'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = stimCountBeginClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    stimCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function stimCountBeginRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'stimCountBegin'-------
    stimCountBeginComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "stimCountBegin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _pj;
var inAllStimComponents;
function inAllStimRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'inAllStim'-------
    t = 0;
    inAllStimClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if ((! _pj.in_es6(stimulus, oldList))) {
        if (((! _pj.in_es6(stimSubCate, oldTargSubList)) && (subCount < setsize))) {
            if ((stimCount === 0)) {
                targetList = (targetList + [stimulus]);
                targetDict[targKey.toString()] = stimulus;
                subCateT[targKey.toString()] = subCate;
                oldList = (oldList + [stimulus]);
                test1Dict[targKey.toString()] = stimulus;
                test2Dict[targKey.toString()] = stimulus;
                test3Dict[targKey.toString()] = stimulus;
                targKey = (targKey + 1);
                stimCount = (stimCount + 1);
            } else {
                if ((stimCount === 1)) {
                    test1Dict[test1Key.toString()] = stimulus;
                    oldList = (oldList + [stimulus]);
                    test1Key = (test1Key + 1);
                    stimCount = (stimCount + 1);
                } else {
                    if ((stimCount === 2)) {
                        test2Dict[test2Key.toString()] = stimulus;
                        oldList = (oldList + [stimulus]);
                        test2Key = (test2Key + 1);
                        stimCount = (stimCount + 1);
                    } else {
                        if ((stimCount === 3)) {
                            test3Dict[test3Key.toString()] = stimulus;
                            oldList = (oldList + [stimulus]);
                            test3Key = (test3Key + 1);
                            stimCount = (stimCount + 1);
                        }
                    }
                }
            }
        } else {
            if ((((! _pj.in_es6(stimulus, oldList)) && (! _pj.in_es6(stimSubCate, oldSubList))) && (stimDcount < 24))) {
                if ((((setsize === 1) || (setsize === 2)) || (setsize === 4))) {
                    if ((stimCount === 0)) {
                        targDistrDictAll[tdAllCount.toString()] = stimulus;
                        subCateTD[tdAllCount.toString()] = subCate;
                        tdAllCount = (tdAllCount + 1);
                        stimCount = (stimCount + 1);
                        stimDcount = (stimDcount + 1);
                    }
                } else {
                    if ((setsize === 8)) {
                        if ((stimDcount < 4)) {
                            if (((stimCount === 0) || (stimCount === 1))) {
                                targDistrDictAll[tdAllCount.toString()] = stimulus;
                                subCateTD[tdAllCount.toString()] = subCate;
                                tdAllCount = (tdAllCount + 1);
                                stimCount = (stimCount + 1);
                                stimDcount = (stimDcount + 1);
                            }
                        } else {
                            if ((stimCount === 0)) {
                                targDistrDictAll[tdAllCount.toString()] = stimulus;
                                subCateTD[tdAllCount.toString()] = subCate;
                                tdAllCount = (tdAllCount + 1);
                                stimCount = (stimCount + 1);
                                stimDcount = (stimDcount + 1);
                            }
                        }
                    } else {
                        if ((setsize === 16)) {
                            if ((stimDcount < 20)) {
                                if (((stimCount === 0) || (stimCount === 1))) {
                                    targDistrDictAll[tdAllCount.toString()] = stimulus;
                                    subCateTD[tdAllCount.toString()] = subCate;
                                    tdAllCount = (tdAllCount + 1);
                                    stimCount = (stimCount + 1);
                                    stimDcount = (stimDcount + 1);
                                }
                            } else {
                                if ((stimCount === 0)) {
                                    targDistrDictAll[tdAllCount.toString()] = stimulus;
                                    subCateTD[tdAllCount.toString()] = subCate;
                                    tdAllCount = (tdAllCount + 1);
                                    stimCount = (stimCount + 1);
                                    stimDcount = (stimDcount + 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // keep track of which components have finished
    inAllStimComponents = [];
    
    inAllStimComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function inAllStimRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'inAllStim'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = inAllStimClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    inAllStimComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function inAllStimRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'inAllStim'-------
    inAllStimComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "inAllStim" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var subCountPlusComponents;
function subCountPlusRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'subCountPlus'-------
    t = 0;
    subCountPlusClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    // keep track of which components have finished
    subCountPlusComponents = [];
    
    subCountPlusComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function subCountPlusRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'subCountPlus'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = subCountPlusClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    subCountPlusComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function subCountPlusRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'subCountPlus'-------
    subCountPlusComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if (((! _pj.in_es6(stimSubCate, oldTargSubList)) && (subCount < setsize))) {
        oldTargSubList = (oldTargSubList + [stimSubCate]);
        subCount = (subCount + 1);
    } else {
        if ((((! _pj.in_es6(stimSubCate, oldSubList)) && (stimDcount < 24)) && (subCount >= setsize))) {
            oldSubList = (oldSubList + [stimSubCate]);
        }
    }
    
    // the Routine "subCountPlus" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var distrCtrlComponents;
function distrCtrlRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'distrCtrl'-------
    t = 0;
    distrCtrlClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    dAllCount = 0;
    
    // keep track of which components have finished
    distrCtrlComponents = [];
    
    distrCtrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function distrCtrlRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'distrCtrl'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = distrCtrlClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    distrCtrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function distrCtrlRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'distrCtrl'-------
    distrCtrlComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "distrCtrl" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var stCount;
var altStimCountBeginComponents;
function altStimCountBeginRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'altStimCountBegin'-------
    t = 0;
    altStimCountBeginClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    stCount = 0;
    
    // keep track of which components have finished
    altStimCountBeginComponents = [];
    
    altStimCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function altStimCountBeginRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'altStimCountBegin'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = altStimCountBeginClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    altStimCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function altStimCountBeginRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'altStimCountBegin'-------
    altStimCountBeginComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "altStimCountBegin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var inAltDistrComponents;
function inAltDistrRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'inAltDistr'-------
    t = 0;
    inAltDistrClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if ((((! _pj.in_es6(stimulus, oldList)) && (stCount === 0)) && (dAllCount < 24))) {
        altDistrDictAll[dAllCount.toString()] = stimulus;
        subCateD[dAllCount.toString()] = subCate;
        dAllCount = (dAllCount + 1);
        stCount = (stCount + 1);
    }
    
    // keep track of which components have finished
    inAltDistrComponents = [];
    
    inAltDistrComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function inAltDistrRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'inAltDistr'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = inAltDistrClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    inAltDistrComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function inAltDistrRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'inAltDistr'-------
    inAltDistrComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "inAltDistr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var distrCount;
var altDistrCount;
var distrCountBeginComponents;
function distrCountBeginRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'distrCountBegin'-------
    t = 0;
    distrCountBeginClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    distrCount = 0;
    altDistrCount = 24;
    
    // keep track of which components have finished
    distrCountBeginComponents = [];
    
    distrCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function distrCountBeginRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'distrCountBegin'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = distrCountBeginClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    distrCountBeginComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function distrCountBeginRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'distrCountBegin'-------
    distrCountBeginComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "distrCountBegin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var inBothDistrComponents;
function inBothDistrRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'inBothDistr'-------
    t = 0;
    inBothDistrClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    searchDict[distrCount.toString()] = targDistrDictAll[distrIndx.toString()];
    searchCate[distrCount.toString()] = cateTag;
    searchSubCate[distrCount.toString()] = subCateTD[distrIndx.toString()];
    oldList = (oldList + [targDistrDictAll[distrIndx.toString()]]);
    searchDict[altDistrCount.toString()] = altDistrDictAll[distrIndx.toString()];
    searchCate[altDistrCount.toString()] = altCateTag;
    searchSubCate[altDistrCount.toString()] = subCateD[distrIndx.toString()];
    oldList = (oldList + [altDistrDictAll[distrIndx.toString()]]);
    distrCount = (distrCount + 1);
    altDistrCount = (altDistrCount + 1);
    
    // keep track of which components have finished
    inBothDistrComponents = [];
    
    inBothDistrComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function inBothDistrRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'inBothDistr'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = inBothDistrClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    inBothDistrComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function inBothDistrRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'inBothDistr'-------
    inBothDistrComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "inBothDistr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var schCount;
var ACClist;
var TBRcount;
var endExpTag;
var stLoopCtrl;
var thLoopCtrl;
var testLoopCtrl;
var memoText;
var testDict;
var testText;
var schPresenLoop;
var ACCschSum;
var ACCrejSum;
var testLoopNum;
var listMergeComponents;
function listMergeRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'listMerge'-------
    t = 0;
    listMergeClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    schCount = 48;
    if ((setsize === 1)) {
        for (var schLoop = 0, _pj_a = 12; (schLoop < _pj_a); schLoop += 1) {
            searchDict[schCount.toString()] = targetDict["0"];
            searchCate[schCount.toString()] = cateTag;
            searchSubCate[schCount.toString()] = subCateT["0"];
            schCount = (schCount + 1);
        }
    } else {
        if ((setsize === 2)) {
            for (var schLoop = 0, _pj_a = 6; (schLoop < _pj_a); schLoop += 1) {
                searchDict[schCount.toString()] = targetDict["0"];
                searchCate[schCount.toString()] = cateTag;
                searchSubCate[schCount.toString()] = subCateT["0"];
                schCount = (schCount + 1);
            }
            for (var schLoop = 6, _pj_a = 12; (schLoop < _pj_a); schLoop += 1) {
                searchDict[schCount.toString()] = targetDict["1"];
                searchCate[schCount.toString()] = cateTag;
                searchSubCate[schCount.toString()] = subCateT["1"];
                schCount = (schCount + 1);
            }
        } else {
            if ((setsize === 4)) {
                for (var schLoop = 0, _pj_a = 3; (schLoop < _pj_a); schLoop += 1) {
                    searchDict[schCount.toString()] = targetDict["0"];
                    searchCate[schCount.toString()] = cateTag;
                    searchSubCate[schCount.toString()] = subCateT["0"];
                    schCount = (schCount + 1);
                }
                for (var schLoop = 3, _pj_a = 6; (schLoop < _pj_a); schLoop += 1) {
                    searchDict[schCount.toString()] = targetDict["1"];
                    searchCate[schCount.toString()] = cateTag;
                    searchSubCate[schCount.toString()] = subCateT["1"];
                    schCount = (schCount + 1);
                }
                for (var schLoop = 6, _pj_a = 9; (schLoop < _pj_a); schLoop += 1) {
                    searchDict[schCount.toString()] = targetDict["2"];
                    searchCate[schCount.toString()] = cateTag;
                    searchSubCate[schCount.toString()] = subCateT["2"];
                    schCount = (schCount + 1);
                }
                for (var schLoop = 9, _pj_a = 12; (schLoop < _pj_a); schLoop += 1) {
                    searchDict[schCount.toString()] = targetDict["3"];
                    searchCate[schCount.toString()] = cateTag;
                    searchSubCate[schCount.toString()] = subCateT["3"];
                    schCount = (schCount + 1);
                }
            } else {
                if ((setsize === 8)) {
                    for (var sizeloop = 0, _pj_a = 8; (sizeloop < _pj_a); sizeloop += 1) {
                        searchDict[schCount.toString()] = targetDict[sizeloop.toString()];
                        searchCate[schCount.toString()] = cateTag;
                        searchSubCate[schCount.toString()] = subCateT[sizeloop.toString()];
                        schCount = (schCount + 1);
                    }
                    for (var sizeloop = 0, _pj_a = 4; (sizeloop < _pj_a); sizeloop += 1) {
                        searchDict[schCount.toString()] = targetDict[sizeloop.toString()];
                        searchCate[schCount.toString()] = cateTag;
                        searchSubCate[schCount.toString()] = subCateT[sizeloop.toString()];
                        schCount = (schCount + 1);
                    }
                } else {
                    if ((setsize === 16)) {
                        for (var sizeloop = 0, _pj_a = 12; (sizeloop < _pj_a); sizeloop += 1) {
                            searchDict[schCount.toString()] = targetDict[sizeloop.toString()];
                            searchCate[schCount.toString()] = cateTag;
                            searchSubCate[schCount.toString()] = subCateT[sizeloop.toString()];
                            schCount = (schCount + 1);
                        }
                    }
                }
            }
        }
    }
    ACClist = {};
    TBRcount = 0;
    endExpTag = 0;
    stLoopCtrl = 1;
    thLoopCtrl = 1;
    testLoopCtrl = 1;
    memoText = "Memorizing";
    testDict = test1Dict;
    testText = "Testing 1";
    schPresenLoop = 1;
    ACCschSum = 0;
    ACCrejSum = 0;
    testLoopNum = testOrd;
    
    // keep track of which components have finished
    listMergeComponents = [];
    
    listMergeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function listMergeRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'listMerge'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = listMergeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    listMergeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function listMergeRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'listMerge'-------
    listMergeComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "listMerge" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var studyCount;
var ACCsum;
var flag;
var TBRLoopCtrlComponents;
function TBRLoopCtrlRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'TBRLoopCtrl'-------
    t = 0;
    TBRLoopCtrlClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    studyCount = 0;
    ACCsum = 0;
    ACCrejSum = 0;
    flag = 0;
    
    // keep track of which components have finished
    TBRLoopCtrlComponents = [];
    
    TBRLoopCtrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function TBRLoopCtrlRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'TBRLoopCtrl'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = TBRLoopCtrlClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    TBRLoopCtrlComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function TBRLoopCtrlRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'TBRLoopCtrl'-------
    TBRLoopCtrlComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "TBRLoopCtrl" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var studyHintComponents;
function studyHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'studyHint'-------
    t = 0;
    studyHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    stHint.setText(memoText);
    // keep track of which components have finished
    studyHintComponents = [];
    studyHintComponents.push(stHint);
    studyHintComponents.push(stFix);
    
    studyHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function studyHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'studyHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = studyHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *stHint* updates
    if (t >= 0.0 && stHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      stHint.tStart = t;  // (not accounting for frame time here)
      stHint.frameNStart = frameN;  // exact frame index
      
      stHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((stHint.status === PsychoJS.Status.STARTED || stHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      stHint.setAutoDraw(false);
    }
    
    // *stFix* updates
    if (t >= 1.2 && stFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      stFix.tStart = t;  // (not accounting for frame time here)
      stFix.frameNStart = frameN;  // exact frame index
      
      stFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((stFix.status === PsychoJS.Status.STARTED || stFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      stFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    studyHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function studyHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'studyHint'-------
    studyHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var studyN;
var studyTrialsComponents;
function studyTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'studyTrials'-------
    t = 0;
    studyTrialsClock.reset(); // clock
    frameN = -1;
    routineTimer.add(3.950000);
    // update component parameters for each repeat
    studyN = studyCount.toString();
    
    studyImg.setImage(targetDict[studyN]);
    // keep track of which components have finished
    studyTrialsComponents = [];
    studyTrialsComponents.push(studyImg);
    studyTrialsComponents.push(studyITI);
    
    studyTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function studyTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'studyTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = studyTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *studyImg* updates
    if (t >= 0.0 && studyImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      studyImg.tStart = t;  // (not accounting for frame time here)
      studyImg.frameNStart = frameN;  // exact frame index
      
      studyImg.setAutoDraw(true);
    }

    frameRemains = 0.0 + 3 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((studyImg.status === PsychoJS.Status.STARTED || studyImg.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      studyImg.setAutoDraw(false);
    }
    
    // *studyITI* updates
    if (t >= 3 && studyITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      studyITI.tStart = t;  // (not accounting for frame time here)
      studyITI.frameNStart = frameN;  // exact frame index
      
      studyITI.setAutoDraw(true);
    }

    frameRemains = 3 + 0.95 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((studyITI.status === PsychoJS.Status.STARTED || studyITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      studyITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    studyTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function studyTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'studyTrials'-------
    studyTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    studyCount = (studyCount + 1);
    
    return Scheduler.Event.NEXT;
  };
}


var testHintComponents;
function testHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'testHint'-------
    t = 0;
    testHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    ttHint.setText(testText);
    ttFix.setText('+');
    // keep track of which components have finished
    testHintComponents = [];
    testHintComponents.push(ttHint);
    testHintComponents.push(ttFix);
    
    testHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function testHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'testHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = testHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *ttHint* updates
    if (t >= 0.0 && ttHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ttHint.tStart = t;  // (not accounting for frame time here)
      ttHint.frameNStart = frameN;  // exact frame index
      
      ttHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ttHint.status === PsychoJS.Status.STARTED || ttHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ttHint.setAutoDraw(false);
    }
    
    // *ttFix* updates
    if (t >= 1.2 && ttFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ttFix.tStart = t;  // (not accounting for frame time here)
      ttFix.frameNStart = frameN;  // exact frame index
      
      ttFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ttFix.status === PsychoJS.Status.STARTED || ttFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ttFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    testHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function testHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'testHint'-------
    testHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var testN;
var testAns;
var _testResp_allKeys;
var testTrialsComponents;
function testTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'testTrials'-------
    t = 0;
    testTrialsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    testN = testIndx.toString();
    if (_pj.in_es6(testDict[testN], targetList)) {
        testAns = "z";
    } else {
        testAns = "m";
    }
    
    testImg.setImage(testDict[testN]);
    testResp.keys = undefined;
    testResp.rt = undefined;
    _testResp_allKeys = [];
    // keep track of which components have finished
    testTrialsComponents = [];
    testTrialsComponents.push(testImg);
    testTrialsComponents.push(testResp);
    
    testTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function testTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'testTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = testTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *testImg* updates
    if (t >= 0.0 && testImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      testImg.tStart = t;  // (not accounting for frame time here)
      testImg.frameNStart = frameN;  // exact frame index
      
      testImg.setAutoDraw(true);
    }

    
    // *testResp* updates
    if (t >= 0.0 && testResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      testResp.tStart = t;  // (not accounting for frame time here)
      testResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { testResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { testResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { testResp.clearEvents(); });
    }

    if (testResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = testResp.getKeys({keyList: ['z', 'm'], waitRelease: false});
      _testResp_allKeys = _testResp_allKeys.concat(theseKeys);
      if (_testResp_allKeys.length > 0) {
        testResp.keys = _testResp_allKeys[0].name;  // just the first key pressed
        testResp.rt = _testResp_allKeys[0].rt;
        // was this correct?
        if (testResp.keys == testAns) {
            testResp.corr = 1;
        } else {
            testResp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    testTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var fbkText;
var clr;
function testTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'testTrials'-------
    testTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // was no response the correct answer?!
    if (testResp.keys === undefined) {
      if (['None','none',undefined].includes(testAns)) {
         testResp.corr = 1;  // correct non-response
      } else {
         testResp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for thisExp (ExperimentHandler)
    psychoJS.experiment.addData('testResp.keys', testResp.keys);
    psychoJS.experiment.addData('testResp.corr', testResp.corr);
    if (typeof testResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('testResp.rt', testResp.rt);
        routineTimer.reset();
        }
    
    testResp.stop();
    if ((testResp.corr === 1)) {
        fbkText = "Correct!";
        clr = "green";
    } else {
        fbkText = "Wrong!";
        clr = "red";
    }
    ACCsum = (ACCsum + testResp.corr);
    test_loop.addData("testsize", setsize);
    test_loop.addData("condTest", cateTag);
    
    // the Routine "testTrials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var test_itiComponents;
function test_itiRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'test_iti'-------
    t = 0;
    test_itiClock.reset(); // clock
    frameN = -1;
    routineTimer.add(1.100000);
    // update component parameters for each repeat
    fbkTest.setColor(new util.Color(clr));
    fbkTest.setText(fbkText);
    // keep track of which components have finished
    test_itiComponents = [];
    test_itiComponents.push(fbkTest);
    test_itiComponents.push(testITI);
    
    test_itiComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function test_itiRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'test_iti'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = test_itiClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fbkTest* updates
    if (t >= 0.0 && fbkTest.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fbkTest.tStart = t;  // (not accounting for frame time here)
      fbkTest.frameNStart = frameN;  // exact frame index
      
      fbkTest.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.6 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((fbkTest.status === PsychoJS.Status.STARTED || fbkTest.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      fbkTest.setAutoDraw(false);
    }
    
    // *testITI* updates
    if (t >= 0.6 && testITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      testITI.tStart = t;  // (not accounting for frame time here)
      testITI.frameNStart = frameN;  // exact frame index
      
      testITI.setAutoDraw(true);
    }

    frameRemains = 0.6 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((testITI.status === PsychoJS.Status.STARTED || testITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      testITI.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    test_itiComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function test_itiRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'test_iti'-------
    test_itiComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var TBRcountPlusComponents;
function TBRcountPlusRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'TBRcountPlus'-------
    t = 0;
    TBRcountPlusClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    ACClist[TBRcount.toString()] = (ACCsum / (setsize * 2));
    TBRcount = (TBRcount + 1);
    if ((TBRcount === 1)) {
        if ((ACClist["0"] >= 0.8)) {
            endExpTag = 0;
            stLoopCtrl = 0;
            thLoopCtrl = 1;
            testLoopCtrl = 1;
            testDict = test2Dict;
            testText = "Testing 2";
        } else {
            if ((ACClist["0"] < 0.8)) {
                endExpTag = 0;
                stLoopCtrl = 1;
                thLoopCtrl = 1;
                testLoopCtrl = 1;
                memoText = "Re-Memorizing";
                testDict = test1Dict;
                testText = "Testing 1";
            }
        }
    } else {
        if ((((TBRcount >= 2) && (ACClist["0"] >= 0.8)) && (ACClist["1"] >= 0.8))) {
            endExpTag = 0;
            stLoopCtrl = 0;
            thLoopCtrl = 0;
            testLoopCtrl = 0;
            flag = 1;
        } else {
            if ((TBRcount === 2)) {
                if (((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8))) {
                    endExpTag = 0;
                    stLoopCtrl = 1;
                    thLoopCtrl = 1;
                    testLoopCtrl = 1;
                    memoText = "Re-Memorizing";
                    testDict = test2Dict;
                    testText = "Testing 2";
                } else {
                    if (((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8))) {
                        endExpTag = 0;
                        stLoopCtrl = 0;
                        thLoopCtrl = 1;
                        testLoopCtrl = 1;
                        testDict = test2Dict;
                        testText = "Testing 2";
                        console.log(stLoopCtrl);
                    } else {
                        if (((ACClist["0"] < 0.8) && (ACClist["1"] < 0.8))) {
                            endExpTag = 0;
                            stLoopCtrl = 1;
                            thLoopCtrl = 1;
                            testLoopCtrl = 1;
                            memoText = "Re-Memorizing";
                            testDict = test1Dict;
                            testText = "Testing 1";
                        }
                    }
                }
            } else {
                if ((TBRcount === 3)) {
                    if (((((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] < 0.8)) || (((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] < 0.8)))) {
                        endExpTag = 0;
                        stLoopCtrl = 1;
                        thLoopCtrl = 1;
                        testLoopCtrl = 1;
                        testDict = test2Dict;
                        testText = "Testing 2";
                    } else {
                        if ((((ACClist["0"] < 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8))) {
                            endExpTag = 0;
                            stLoopCtrl = 0;
                            thLoopCtrl = 1;
                            testLoopCtrl = 1;
                            testDict = test2Dict;
                            testText = "Testing 2";
                        } else {
                            if (((((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] >= 0.8)) || (((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8)))) {
                                endExpTag = 0;
                                stLoopCtrl = 0;
                                thLoopCtrl = 1;
                                testLoopCtrl = 1;
                                testDict = test3Dict;
                                testText = "Testing 3";
                            } else {
                                if ((((ACClist["0"] < 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] < 0.8))) {
                                    endExpTag = 1;
                                    stLoopCtrl = 0;
                                    thLoopCtrl = 0;
                                    testLoopCtrl = 0;
                                    schPresenLoop = 0;
                                }
                            }
                        }
                    }
                } else {
                    if (((TBRcount >= 4) && (((((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] >= 0.8)) || ((((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] >= 0.8))))) {
                        endExpTag = 0;
                        stLoopCtrl = 0;
                        thLoopCtrl = 0;
                        testLoopCtrl = 0;
                        flag = 1;
                    } else {
                        if ((TBRcount === 4)) {
                            if (((((((ACClist["0"] < 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] >= 0.8)) || ((((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] < 0.8)) && (ACClist["3"] >= 0.8))) || ((((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] < 0.8)) && (ACClist["3"] >= 0.8)))) {
                                endExpTag = 0;
                                stLoopCtrl = 0;
                                thLoopCtrl = 1;
                                testLoopCtrl = 1;
                                testDict = test3Dict;
                                testText = "Testing 3";
                            } else {
                                if ((((((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] < 0.8)) || ((((ACClist["0"] > 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] < 0.8)))) {
                                    endExpTag = 0;
                                    stLoopCtrl = 1;
                                    thLoopCtrl = 1;
                                    testLoopCtrl = 1;
                                    testDict = test3Dict;
                                    testText = "Testing 3";
                                } else {
                                    if (((((((ACClist["0"] >= 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] < 0.8)) && (ACClist["3"] < 0.8)) || ((((ACClist["0"] < 0.8) && (ACClist["1"] >= 0.8)) && (ACClist["2"] < 0.8)) && (ACClist["3"] < 0.8))) || ((((ACClist["0"] < 0.8) && (ACClist["1"] < 0.8)) && (ACClist["2"] >= 0.8)) && (ACClist["3"] < 0.8)))) {
                                        endExpTag = 1;
                                        stLoopCtrl = 0;
                                        thLoopCtrl = 0;
                                        testLoopCtrl = 0;
                                        schPresenLoop = 0;
                                    }
                                }
                            }
                        } else {
                            if (((TBRcount === 5) && (ACClist["4"] < 0.8))) {
                                endExpTag = 0;
                                stLoopCtrl = 0;
                                thLoopCtrl = 0;
                                testLoopCtrl = 0;
                            } else {
                                if (((TBRcount === 5) && (ACClist["4"] >= 0.8))) {
                                    endExpTag = 0;
                                    stLoopCtrl = 0;
                                    thLoopCtrl = 0;
                                    testLoopCtrl = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // keep track of which components have finished
    TBRcountPlusComponents = [];
    
    TBRcountPlusComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function TBRcountPlusRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'TBRcountPlus'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = TBRcountPlusClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    TBRcountPlusComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function TBRcountPlusRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'TBRcountPlus'-------
    TBRcountPlusComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "TBRcountPlus" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var expENDComponents;
function expENDRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'expEND'-------
    t = 0;
    expENDClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    // keep track of which components have finished
    expENDComponents = [];
    expENDComponents.push(expENDtest);
    
    expENDComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function expENDRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'expEND'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = expENDClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *expENDtest* updates
    if (t >= 0.0 && expENDtest.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      expENDtest.tStart = t;  // (not accounting for frame time here)
      expENDtest.frameNStart = frameN;  // exact frame index
      
      expENDtest.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    expENDComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function expENDRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'expEND'-------
    expENDComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // the Routine "expEND" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var searchHintComponents;
function searchHintRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'searchHint'-------
    t = 0;
    searchHintClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    searchHintComponents = [];
    searchHintComponents.push(schHint);
    searchHintComponents.push(schFix);
    
    searchHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function searchHintRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'searchHint'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = searchHintClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *schHint* updates
    if (t >= 0.0 && schHint.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      schHint.tStart = t;  // (not accounting for frame time here)
      schHint.frameNStart = frameN;  // exact frame index
      
      schHint.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((schHint.status === PsychoJS.Status.STARTED || schHint.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      schHint.setAutoDraw(false);
    }
    
    // *schFix* updates
    if (t >= 1.2 && schFix.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      schFix.tStart = t;  // (not accounting for frame time here)
      schFix.frameNStart = frameN;  // exact frame index
      
      schFix.setAutoDraw(true);
    }

    frameRemains = 1.2 + 0.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((schFix.status === PsychoJS.Status.STARTED || schFix.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      schFix.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    searchHintComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function searchHintRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'searchHint'-------
    searchHintComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


var schN;
var schAns;
var _schResp_allKeys;
var searchTrialsComponents;
function searchTrialsRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'searchTrials'-------
    t = 0;
    searchTrialsClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    schN = trialIndx.toString();
    if (_pj.in_es6(searchDict[schN], targetList)) {
        schAns = "z";
    } else {
        schAns = "m";
    }
    
    schImg.setImage(searchDict[schN]);
    schResp.keys = undefined;
    schResp.rt = undefined;
    _schResp_allKeys = [];
    // keep track of which components have finished
    searchTrialsComponents = [];
    searchTrialsComponents.push(schImg);
    searchTrialsComponents.push(schITI);
    searchTrialsComponents.push(schResp);
    
    searchTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function searchTrialsRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'searchTrials'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = searchTrialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *schImg* updates
    if (t >= 0.0 && schImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      schImg.tStart = t;  // (not accounting for frame time here)
      schImg.frameNStart = frameN;  // exact frame index
      
      schImg.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((schImg.status === PsychoJS.Status.STARTED || schImg.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      schImg.setAutoDraw(false);
    }
    
    // *schITI* updates
    if (t >= 0.2 && schITI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      schITI.tStart = t;  // (not accounting for frame time here)
      schITI.frameNStart = frameN;  // exact frame index
      
      schITI.setAutoDraw(true);
    }

    frameRemains = 0.2 + 1.8 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((schITI.status === PsychoJS.Status.STARTED || schITI.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      schITI.setAutoDraw(false);
    }
    
    // *schResp* updates
    if (t >= 0.0 && schResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      schResp.tStart = t;  // (not accounting for frame time here)
      schResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { schResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { schResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { schResp.clearEvents(); });
    }

    frameRemains = 0.0 + 2.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((schResp.status === PsychoJS.Status.STARTED || schResp.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      schResp.status = PsychoJS.Status.FINISHED;
  }

    if (schResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = schResp.getKeys({keyList: ['z', 'm'], waitRelease: false});
      _schResp_allKeys = _schResp_allKeys.concat(theseKeys);
      if (_schResp_allKeys.length > 0) {
        schResp.keys = _schResp_allKeys[0].name;  // just the first key pressed
        schResp.rt = _schResp_allKeys[0].rt;
        // was this correct?
        if (schResp.keys == schAns) {
            schResp.corr = 1;
        } else {
            schResp.corr = 0;
        }
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    searchTrialsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function searchTrialsRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'searchTrials'-------
    searchTrialsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // was no response the correct answer?!
    if (schResp.keys === undefined) {
      if (['None','none',undefined].includes(schAns)) {
         schResp.corr = 1;  // correct non-response
      } else {
         schResp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for thisExp (ExperimentHandler)
    psychoJS.experiment.addData('schResp.keys', schResp.keys);
    psychoJS.experiment.addData('schResp.corr', schResp.corr);
    if (typeof schResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('schResp.rt', schResp.rt);
        }
    
    schResp.stop();
    if (((schResp.corr === 1) && (schResp.keys === "z"))) {
        ACCschSum = (ACCschSum + 1);
    } else {
        if (((schResp.corr === 1) && (schResp.keys === "m"))) {
            ACCrejSum = (ACCrejSum + 1);
        }
    }
    sch_loop.addData("setsize", setsize);
    sch_loop.addData("subCategory", searchSubCate[schN]);
    sch_loop.addData("category", searchCate[schN]);
    sch_loop.addData("cond", cateTag);
    sch_loop.addData("stimImg", searchDict[schN]);
    
    return Scheduler.Event.NEXT;
  };
}


var ACCsch;
var ACCschStr;
var ACCstr;
var ACCtext1;
var ACCschR;
var ACCschStrR;
var ACCstrR;
var ACCtext2;
var _brkResp_allKeys;
var schACCpresenComponents;
function schACCpresenRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'schACCpresen'-------
    t = 0;
    schACCpresenClock.reset(); // clock
    frameN = -1;
    routineTimer.add(300.000000);
    // update component parameters for each repeat
    ACCsch = ((ACCschSum / targetN) * 100);
    ACCschStr = ACCsch.toString();
    ACCstr = "";
    if ((ACCschStr.length > 5)) {
        for (var aN = 0, _pj_a = 5; (aN < _pj_a); aN += 1) {
            ACCstr = (ACCstr + ACCschStr[aN]);
        }
    } else {
        for (var aN = 0, _pj_a = ACCschStr.length; (aN < _pj_a); aN += 1) {
            ACCstr = (ACCstr + ACCschStr[aN]);
        }
    }
    ACCtext1 = (("You detected " + ACCstr) + "% of the targets");
    ACCschR = ((ACCrejSum / (distrN * 2)) * 100);
    ACCschStrR = ACCschR.toString();
    ACCstrR = "";
    if ((ACCschStrR.length > 5)) {
        for (var aN = 0, _pj_a = 5; (aN < _pj_a); aN += 1) {
            ACCstrR = (ACCstrR + ACCschStrR[aN]);
        }
    } else {
        for (var aN = 0, _pj_a = ACCschStrR.length; (aN < _pj_a); aN += 1) {
            ACCstrR = (ACCstrR + ACCschStrR[aN]);
        }
    }
    ACCtext2 = (("You rejected " + ACCstrR) + "% of the distractors");
    
    ACCpresen1.setText(ACCtext1);
    ACCpresen2.setText(ACCtext2);
    brkResp.keys = undefined;
    brkResp.rt = undefined;
    _brkResp_allKeys = [];
    // keep track of which components have finished
    schACCpresenComponents = [];
    schACCpresenComponents.push(ACCpresen1);
    schACCpresenComponents.push(ACCpresen2);
    schACCpresenComponents.push(ACCpresen3);
    schACCpresenComponents.push(ACCpresen4);
    schACCpresenComponents.push(ACCpresen5);
    schACCpresenComponents.push(ACCpresen6);
    schACCpresenComponents.push(brkResp);
    
    schACCpresenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function schACCpresenRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'schACCpresen'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = schACCpresenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *ACCpresen1* updates
    if (t >= 0.0 && ACCpresen1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen1.tStart = t;  // (not accounting for frame time here)
      ACCpresen1.frameNStart = frameN;  // exact frame index
      
      ACCpresen1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen1.status === PsychoJS.Status.STARTED || ACCpresen1.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen1.setAutoDraw(false);
    }
    
    // *ACCpresen2* updates
    if (t >= 0.0 && ACCpresen2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen2.tStart = t;  // (not accounting for frame time here)
      ACCpresen2.frameNStart = frameN;  // exact frame index
      
      ACCpresen2.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen2.status === PsychoJS.Status.STARTED || ACCpresen2.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen2.setAutoDraw(false);
    }
    
    // *ACCpresen3* updates
    if (t >= 0.0 && ACCpresen3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen3.tStart = t;  // (not accounting for frame time here)
      ACCpresen3.frameNStart = frameN;  // exact frame index
      
      ACCpresen3.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen3.status === PsychoJS.Status.STARTED || ACCpresen3.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen3.setAutoDraw(false);
    }
    
    // *ACCpresen4* updates
    if (t >= 0.0 && ACCpresen4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen4.tStart = t;  // (not accounting for frame time here)
      ACCpresen4.frameNStart = frameN;  // exact frame index
      
      ACCpresen4.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen4.status === PsychoJS.Status.STARTED || ACCpresen4.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen4.setAutoDraw(false);
    }
    
    // *ACCpresen5* updates
    if (t >= 0.0 && ACCpresen5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen5.tStart = t;  // (not accounting for frame time here)
      ACCpresen5.frameNStart = frameN;  // exact frame index
      
      ACCpresen5.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen5.status === PsychoJS.Status.STARTED || ACCpresen5.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen5.setAutoDraw(false);
    }
    
    // *ACCpresen6* updates
    if (t >= 0.0 && ACCpresen6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ACCpresen6.tStart = t;  // (not accounting for frame time here)
      ACCpresen6.frameNStart = frameN;  // exact frame index
      
      ACCpresen6.setAutoDraw(true);
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((ACCpresen6.status === PsychoJS.Status.STARTED || ACCpresen6.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      ACCpresen6.setAutoDraw(false);
    }
    
    // *brkResp* updates
    if (t >= 0.0 && brkResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      brkResp.tStart = t;  // (not accounting for frame time here)
      brkResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { brkResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { brkResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { brkResp.clearEvents(); });
    }

    frameRemains = 0.0 + 300 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((brkResp.status === PsychoJS.Status.STARTED || brkResp.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      brkResp.status = PsychoJS.Status.FINISHED;
  }

    if (brkResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = brkResp.getKeys({keyList: ['space'], waitRelease: false});
      _brkResp_allKeys = _brkResp_allKeys.concat(theseKeys);
      if (_brkResp_allKeys.length > 0) {
        brkResp.keys = _brkResp_allKeys[_brkResp_allKeys.length - 1].name;  // just the last key pressed
        brkResp.rt = _brkResp_allKeys[_brkResp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    schACCpresenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function schACCpresenRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'schACCpresen'-------
    schACCpresenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('brkResp.keys', brkResp.keys);
    if (typeof brkResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('brkResp.rt', brkResp.rt);
        routineTimer.reset();
        }
    
    brkResp.stop();
    return Scheduler.Event.NEXT;
  };
}


var goodbyeComponents;
function goodbyeRoutineBegin(snapshot) {
  return function () {
    //------Prepare to start Routine 'goodbye'-------
    t = 0;
    goodbyeClock.reset(); // clock
    frameN = -1;
    routineTimer.add(3.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    goodbyeComponents = [];
    goodbyeComponents.push(gdy);
    
    goodbyeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
  };
}


function goodbyeRoutineEachFrame(snapshot) {
  return function () {
    //------Loop for each frame of Routine 'goodbye'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = goodbyeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *gdy* updates
    if (t >= 0.0 && gdy.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      gdy.tStart = t;  // (not accounting for frame time here)
      gdy.frameNStart = frameN;  // exact frame index
      
      gdy.setAutoDraw(true);
    }

    frameRemains = 0.0 + 3 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if ((gdy.status === PsychoJS.Status.STARTED || gdy.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      gdy.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    goodbyeComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function goodbyeRoutineEnd(snapshot) {
  return function () {
    //------Ending Routine 'goodbye'-------
    goodbyeComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        const thisTrial = snapshot.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(snapshot);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(currentLoop) {
  return function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
