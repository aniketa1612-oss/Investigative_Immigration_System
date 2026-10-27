import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { request } from './api';

const FALLBACK_LANGUAGES = [{name:'English',code:'en',locale:'en-US'},{name:'Hindi',code:'hi',locale:'hi-IN'}];

function Badge({children, tone='teal'}) { return <span className={`badge ${tone}`}>{children}</span>; }
function Section({title, children, className=''}) { return <section className={`panel ${className}`}><div className="section-title"><h2>{title}</h2></div>{children}</section>; }

// Keep local model requests in statement order without holding up audio uploads.
let translationQueue = Promise.resolve();
function StatementTranslation({interviewId, statement, targetLanguage}) {
  const [result, setResult] = useState(null);
  const [failure, setFailure] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setResult(null); setFailure('');
    if (statement.language === targetLanguage) return;
    translationQueue = translationQueue.catch(() => {}).then(async () => {
      if (cancelled) return;
      try {
        const translated = await request(`/api/interviews/${interviewId}/transcripts/${statement.id}/translation?target_language=${encodeURIComponent(targetLanguage)}`, {method:'POST'});
        if (!cancelled) setResult(translated);
      } catch (err) { if (!cancelled) setFailure(err.message); }
    });
    return () => { cancelled = true; };
  }, [interviewId, statement.id, statement.text, statement.language, targetLanguage, attempt]);
  if (statement.language === targetLanguage) return null;
  return <div className="statement-translation" aria-live="polite">
    <strong>{targetLanguage} · Machine translation</strong>
    {result ? <><p dir="auto">{result.text}</p>{result.note && <p className="translation-note">Translation note: {result.note}</p>}</>
      : failure ? <><p role="alert">Translation unavailable: {failure}</p><button onClick={()=>setAttempt(value=>value+1)}>Retry translation</button></>
      : <p role="status">Waiting for translation…</p>}
  </div>;
}

function formatDate(value) {
  if (!value) return 'Not recorded';
  return new Date(value).toLocaleString();
}

function formatDuration(seconds) {
  if (seconds == null) return 'Not available';
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
}

function ReportList({items}) {
  return <ul className="report-list">{items?.map((item, index)=><li key={`${index}-${item}`}>{item}</li>)}</ul>;
}

function BarChart({data, valueLabel}) {
  const max = Math.max(...data.map(item=>item.value), 1);
  if (!data.length) return <p className="empty">No recorded data is available for this chart.</p>;
  return <div className="bar-chart">{data.map(item=><div className="bar-row" key={item.label}>
    <span>{item.label}</span>
    <div className="bar-track"><i style={{width:`${Math.max(3, item.value / max * 100)}%`}}/></div>
    <strong>{item.value}{valueLabel}</strong>
  </div>)}</div>;
}

function StrengthChart({data}) {
  if (!data.length) return <p className="empty">No facial activity observations were recorded.</p>;
  const width = 640, height = 180, inset = 24;
  const points = data.map((item,index)=>{
    const x = data.length === 1 ? width / 2 : inset + index * (width - inset * 2) / (data.length - 1);
    const y = height - inset - item.value * (height - inset * 2);
    return {x,y,...item};
  });
  return <div className="line-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recorded facial movement detection strength by observation">
      {[0,.25,.5,.75,1].map(level=><line key={level} x1={inset} x2={width-inset} y1={height-inset-level*(height-inset*2)} y2={height-inset-level*(height-inset*2)} />)}
      <polyline points={points.map(point=>`${point.x},${point.y}`).join(' ')} />
      {points.map(point=><circle key={point.label} cx={point.x} cy={point.y} r="5"><title>{point.label}: {Math.round(point.value*100)}% — {point.pattern}</title></circle>)}
    </svg>
    <div className="chart-scale"><span>0%</span><span>Detection strength</span><span>100%</span></div>
  </div>;
}

export default function App() {
  const [step, setStep] = useState('setup');
  const [consent, setConsent] = useState(false);
  const [language, setLanguage] = useState('English');
  const [languages, setLanguages] = useState(FALLBACK_LANGUAGES);
  const [officerLanguage, setOfficerLanguage] = useState('English');
  const [translationEnabled, setTranslationEnabled] = useState(true);
  const [interviewSetting, setInterviewSetting] = useState('airport');
  const [intake, setIntake] = useState({full_name:'',age:'',relationship_to_event:'',contact_reference:'',initial_statement:'',additional_notes:''});
  const [interview, setInterview] = useState(null);
  const [status, setStatus] = useState('created');
  const [error, setError] = useState('');
  const [camera, setCamera] = useState(false);
  const [mic, setMic] = useState(false);
  const [checkingDevices, setCheckingDevices] = useState(false);
  const [lifecyclePending, setLifecyclePending] = useState(false);
  const [listening, setListening] = useState(false);
  const [interimText, setInterimText] = useState('');
  const [transcripts, setTranscripts] = useState([]);
  const [manualText, setManualText] = useState('');
  const [question, setQuestion] = useState(null);
  const [questionPending, setQuestionPending] = useState(false);
  const [report, setReport] = useState(null);
  const [reportPending, setReportPending] = useState(false);
  const [faceAnalysis, setFaceAnalysis] = useState(null);
  const [faceAnalysisActive, setFaceAnalysisActive] = useState(false);
  const [faceAnalysisLoading, setFaceAnalysisLoading] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const recorderChunksRef = useRef([]);
  const recorderTimerRef = useRef(null);
  const uploadChainRef = useRef(Promise.resolve());
  const shouldListenRef = useRef(false);
  const speechTimerRef = useRef(null);
  const speechWatchdogRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const faceFrameRef = useRef(null);
  const lastFaceFrameRef = useRef(0);
  const lastFaceStoreRef = useRef({pattern:'', time:0});

  useEffect(() => () => {
    shouldListenRef.current = false;
    recorderRef.current?.stop();
    clearTimeout(recorderTimerRef.current);
    clearTimeout(speechTimerRef.current);
    clearTimeout(speechWatchdogRef.current);
    window.speechSynthesis?.cancel();
    cancelAnimationFrame(faceFrameRef.current);
    faceLandmarkerRef.current?.close();
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  useEffect(() => {
    request('/api/config').then(config => {
      if (config.languages?.length) setLanguages(config.languages);
    }).catch(() => {});
  }, []);

  async function createInterview(event) {
    event.preventDefault(); setError('');
    try {
      const interviewee = {...intake, age:intake.age ? Number(intake.age) : null, contact_reference:intake.contact_reference || null, additional_notes:intake.additional_notes || null};
      const settingLabel = interviewSetting === 'airport' ? 'Airport or border-entry immigration interview' : 'Visa-office immigration interview';
      const objective = interviewSetting === 'airport'
        ? 'Conduct a neutral airport or border-entry immigration interview covering travel purpose, itinerary, accommodation, onward or return travel, documents, and declarations without making an admissibility decision.'
        : 'Conduct a neutral visa-office immigration interview covering application purpose, intended activities, itinerary, accommodation, funding, supporting documents, and application history without making an eligibility decision.';
      const created = await request('/api/interviews', {method:'POST', body:JSON.stringify({title:`${settingLabel} · ${new Date().toLocaleString()}`, objective, language, consent_given:consent, raw_media_opt_in:false, interviewee})});
      setInterview(created); setStatus(created.status); setStep('live');
      setTranscripts(await request(`/api/interviews/${created.id}/transcripts`));
    } catch (err) { setError(err.message); }
  }

  async function checkDevices() {
    setError(''); setCheckingDevices(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
      stopDevices();
      // Acquire the microphone independently. A missing or blocked camera must
      // not prevent speech transcription from working.
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true},
        video:false,
      });
      let videoStream = null;
      try {
        videoStream = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
      } catch {
        setError('Microphone is active, but the camera is unavailable. Speech transcription will still work.');
      }
      const stream = new MediaStream([
        ...audioStream.getAudioTracks(),
        ...(videoStream?.getVideoTracks() || []),
      ]);
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      syncDeviceState(stream);
      stream.getTracks().forEach(track => track.addEventListener('ended', () => syncDeviceState(stream)));
      return audioTracksAreLive(stream);
    } catch (err) {
      stopDevices();
      setError(err?.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow microphone access in browser site settings.' : err?.name === 'NotFoundError' ? 'No usable microphone was found.' : 'The microphone could not be started. Close other apps using it and try again.');
      return false;
    } finally { setCheckingDevices(false); }
  }

  function audioTracksAreLive(stream = streamRef.current) {
    return Boolean(stream?.getAudioTracks().some(track => track.readyState === 'live'));
  }

  function syncDeviceState(stream = streamRef.current) {
    const tracks = stream?.getTracks() || [];
    setCamera(tracks.some(track => track.kind === 'video' && track.readyState === 'live'));
    setMic(tracks.some(track => track.kind === 'audio' && track.readyState === 'live'));
  }

  function stopDevices() {
    stopFaceAnalysis();
    stopListening();
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamera(false); setMic(false);
  }

  function interpretFaceBlendshapes(categories) {
    const scores = Object.fromEntries(categories.map(item=>[item.categoryName, item.score]));
    const average = (...names) => names.reduce((sum,name)=>sum+(scores[name] || 0),0)/names.length;
    const expressionScores = {
      'Happiness-like': Math.min(1, average('mouthSmileLeft','mouthSmileRight')*.7 + average('cheekSquintLeft','cheekSquintRight')*.3),
      'Surprise-like': Math.min(1, average('eyeWideLeft','eyeWideRight')*.4 + (scores.browInnerUp || 0)*.3 + (scores.jawOpen || 0)*.3),
      'Sadness-like': Math.min(1, average('mouthFrownLeft','mouthFrownRight')*.55 + (scores.browInnerUp || 0)*.45),
      'Anger-like': Math.min(1, average('browDownLeft','browDownRight')*.45 + average('eyeSquintLeft','eyeSquintRight')*.3 + average('mouthPressLeft','mouthPressRight')*.25),
      'Disgust-like': Math.min(1, average('noseSneerLeft','noseSneerRight')*.65 + (scores.mouthUpperUpLeft || 0)*.175 + (scores.mouthUpperUpRight || 0)*.175),
    };
    const [candidateExpression, candidateScore] = Object.entries(expressionScores).sort((left,right)=>right[1]-left[1])[0];
    const expression = candidateScore >= .32 ? candidateExpression : 'Neutral or undetermined';
    const expressionConfidence = candidateScore >= .32 ? candidateScore : Math.max(0, 1-candidateScore);
    const active = categories.filter(item => item.categoryName !== '_neutral' && item.score >= .25).sort((a,b) => b.score-a.score).slice(0,3);
    if (!active.length) return {face_detected:true, pattern:'No prominent facial movement', cues:[], strength:0, expression:'Neutral or undetermined', expressionConfidence:Number(expressionConfidence.toFixed(2)), expressionScores, interpretation:'No prominent facial movement was detected; the expression estimate is neutral or undetermined.'};
    const dominant = active[0];
    const name = dominant.categoryName;
    const pattern = /smile/i.test(name) ? 'Smile-related mouth movement' : /brow/i.test(name) ? 'Brow movement' : /eye|blink/i.test(name) ? 'Eye or blink movement' : /jaw|mouth/i.test(name) ? 'Mouth or jaw movement' : 'Facial movement';
    return {face_detected:true, pattern, cues:active.map(item=>({name:item.categoryName, strength:Number(item.score.toFixed(2))})), strength:Number(dominant.score.toFixed(2)), expression, expressionConfidence:Number(expressionConfidence.toFixed(2)), expressionScores, interpretation:`${pattern} was observed. The cue combination is most consistent with a ${expression.toLowerCase()} expression, but camera-based expression estimates can be wrong and do not establish the person’s internal emotion.`};
  }

  async function storeFaceObservation(result) {
    const now = Date.now();
    const previous = lastFaceStoreRef.current;
    if (result.pattern === previous.pattern && now-previous.time < 10000) return;
    if (now-previous.time < 5000) return;
    lastFaceStoreRef.current = {pattern:result.pattern,time:now};
    try {
      await request(`/api/interviews/${interview.id}/facial-observations`, {method:'POST',body:JSON.stringify({observed:{pattern:result.pattern,cues:result.cues},interpreted:{summary:result.interpretation,scope:'facial-expression-estimate',expression_estimate:result.expression,expression_confidence:result.expressionConfidence,expression_scores:result.expressionScores},confidence:result.strength})});
    } catch (err) { setError(err.message); }
  }

  async function startFaceAnalysis() {
    if (!streamRef.current?.getVideoTracks().some(track=>track.readyState==='live') || faceAnalysisActive || faceAnalysisLoading) return;
    setFaceAnalysisLoading(true);
    try {
      if (!faceLandmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks('/models/mediapipe/wasm');
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/models/mediapipe/face_landmarker.task'},runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true});
      }
      setFaceAnalysisActive(true);
      const analyseFrame = () => {
        const video = videoRef.current;
        if (!video || video.readyState < 2 || !streamRef.current) { faceFrameRef.current=requestAnimationFrame(analyseFrame); return; }
        if (performance.now()-lastFaceFrameRef.current < 350) { faceFrameRef.current=requestAnimationFrame(analyseFrame); return; }
        lastFaceFrameRef.current=performance.now();
        const detected = faceLandmarkerRef.current.detectForVideo(video, performance.now());
        if (detected.faceBlendshapes?.length) {
          const result = interpretFaceBlendshapes(detected.faceBlendshapes[0].categories);
          setFaceAnalysis(result); storeFaceObservation(result);
        } else setFaceAnalysis({face_detected:false,pattern:'No face detected',cues:[],strength:0,expression:'Unavailable',expressionConfidence:0,interpretation:'Position one face clearly within the camera frame.'});
        faceFrameRef.current=requestAnimationFrame(analyseFrame);
      };
      faceFrameRef.current=requestAnimationFrame(analyseFrame);
    } catch { setError('Facial activity analysis could not start. Run the offline model preparation script and restart the frontend.'); }
    finally { setFaceAnalysisLoading(false); }
  }

  function stopFaceAnalysis() {
    cancelAnimationFrame(faceFrameRef.current);
    faceFrameRef.current=null;
    setFaceAnalysisActive(false);
  }

  function stopListening() {
    shouldListenRef.current = false;
    clearTimeout(recorderTimerRef.current);
    if (recorderRef.current?.state !== 'inactive') recorderRef.current?.stop();
    recorderRef.current = null;
    setListening(false); setInterimText('');
  }

  async function saveTranscript(text, confidence=1) {
    const clean = text.trim();
    if (!clean) return false;
    try {
      const saved = await request(`/api/interviews/${interview.id}/transcripts`, {method:'POST', body:JSON.stringify({speaker:'interviewee', text:clean, language, confidence})});
      setTranscripts(current => [...current, saved]);
      return true;
    } catch (err) { setError(err.message); return false; }
  }

  function startRecordingCycle() {
    if (!shouldListenRef.current || !audioTracksAreLive()) return;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') return;
    const audioStream = new MediaStream(streamRef.current.getAudioTracks());
    const mimeType = ['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(type=>MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(audioStream, mimeType ? {mimeType} : undefined);
    recorderChunksRef.current = [];
    recorder.ondataavailable = event => { if (event.data.size) recorderChunksRef.current.push(event.data); };
    recorder.onerror = () => setError('Local audio capture failed. You can enter the statement manually.');
    recorder.onstop = () => {
      const blob = new Blob(recorderChunksRef.current, {type:recorder.mimeType || 'audio/webm'});
      if (blob.size > 1000) {
        setInterimText('Transcribing the latest audio locally…');
        uploadChainRef.current = uploadChainRef.current.then(async () => {
          const form = new FormData();
          form.append('language', language);
          form.append('audio', blob, recorder.mimeType.includes('mp4') ? 'segment.mp4' : 'segment.webm');
          const saved = await request(`/api/interviews/${interview.id}/audio-segments`, {method:'POST', body:form});
          if (saved.saved) setTranscripts(current => current.some(item=>item.id===saved.id) ? current : [...current, saved]);
          else setError('No speech was detected in that audio segment. Speak closer to the microphone and try again.');
        }).catch(err => setError(`Transcription failed: ${err.message}`)).finally(() => setInterimText(''));
      }
      if (shouldListenRef.current) startRecordingCycle();
    };
    recorderRef.current = recorder;
    recorder.start();
    recorderTimerRef.current = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 8000);
  }

  function startListening() {
    setError('');
    if (!window.MediaRecorder) { setError('Audio recording is not supported in this browser. Use a current browser or enter the statement manually.'); return; }
    if (!audioTracksAreLive()) { setError('Check and allow the microphone before starting transcription.'); return; }
    shouldListenRef.current = true;
    setListening(true);
    startRecordingCycle();
  }

  function speakQuestion(text, resumeTranscription = shouldListenRef.current) {
    if (!text) return;
    const speech = window.speechSynthesis;
    if (!speech || !window.SpeechSynthesisUtterance) {
      if (resumeTranscription) startListening();
      return;
    }
    stopListening();
    clearTimeout(speechTimerRef.current);
    clearTimeout(speechWatchdogRef.current);
    speech.cancel();

    let finished = false;
    let speakingStarted = false;
    const resume = () => {
      if (finished) return;
      finished = true;
      clearTimeout(speechTimerRef.current);
      clearTimeout(speechWatchdogRef.current);
      if (resumeTranscription && audioTracksAreLive()) startListening();
    };
    const speak = () => {
      if (finished || speakingStarted) return;
      speakingStarted = true;
      const utterance = new SpeechSynthesisUtterance(text);
      const locale = languages.find(item=>item.name===language)?.locale || 'en-US';
      utterance.lang = locale;
      const voices = speech.getVoices();
      const exactVoice = voices.find(voice => voice.lang.toLowerCase() === locale.toLowerCase());
      const languageVoice = voices.find(voice => voice.lang.toLowerCase().startsWith(locale.split('-')[0].toLowerCase()));
      utterance.voice = exactVoice || languageVoice || null;
      utterance.onend = resume;
      utterance.onerror = () => {
        setError(`The browser could not read the ${language} question aloud. Use “Read question aloud” again or install a ${language} system voice.`);
        resume();
      };
      speech.speak(utterance);
      speechTimerRef.current = setTimeout(() => {
        if (!speech.speaking && !speech.pending) {
          setError(`No ${language} speech voice started. The transcript microphone has been resumed.`);
          resume();
        }
      }, 1800);
      // Some browser/voice combinations never emit `onend`. Resume capture
      // after a conservative maximum so transcription cannot remain stuck off.
      const expectedDuration = Math.min(45000, Math.max(9000, text.trim().split(/\s+/).length * 700));
      speechWatchdogRef.current = setTimeout(() => {
        if (!finished) {
          speech.cancel();
          setError('Question playback did not finish cleanly, so microphone transcription was resumed automatically.');
          resume();
        }
      }, expectedDuration);
    };

    if (speech.getVoices().length) speak();
    else {
      const voiceTimeout = setTimeout(speak, 1000);
      speech.addEventListener('voiceschanged', () => { clearTimeout(voiceTimeout); speak(); }, {once:true});
    }
  }

  async function saveManualTranscript() {
    if (!manualText.trim()) return;
    if (await saveTranscript(manualText)) setManualText('');
  }

  async function action(name) {
    if (lifecyclePending) return;
    setError(''); setLifecyclePending(true);
    try {
      let microphoneReady = audioTracksAreLive();
      if (name === 'start' && !microphoneReady) microphoneReady = await checkDevices();
      const result = await request(`/api/interviews/${interview.id}/lifecycle/${name}`, {method:'POST'});
      setStatus(result.status);
      if (name === 'start' && !question) {
        const opening = await request(`/api/interviews/${interview.id}/questions`, {method:'POST', body:JSON.stringify({language})});
        setQuestion(opening.text);
        speakQuestion(opening.text, microphoneReady);
      }
      if (name === 'start') startFaceAnalysis();
      if (name === 'resume') { if (microphoneReady) startListening(); startFaceAnalysis(); }
      if (name === 'pause') { window.speechSynthesis?.cancel(); stopListening(); stopFaceAnalysis(); }
      if (name === 'end') { window.speechSynthesis?.cancel(); stopDevices(); }
    } catch (err) { setError(err.message); }
    finally { setLifecyclePending(false); }
  }

  async function generateQuestion() {
    if ((!transcripts.length && !manualText.trim()) || questionPending) return;
    setError(''); setQuestionPending(true);
    try {
      if (manualText.trim()) {
        const saved = await saveTranscript(manualText);
        if (!saved) return;
        setManualText('');
      }
      const result = await request(`/api/interviews/${interview.id}/questions`, {method:'POST', body:JSON.stringify({language})});
      setQuestion(result.text);
      speakQuestion(result.text);
    } catch (err) { setError(err.message); }
    finally { setQuestionPending(false); }
  }

  async function makeReport() {
    if (reportPending) return;
    setError(''); setReportPending(true);
    try { setReport(await request(`/api/interviews/${interview.id}/report`, {method:'POST'})); setStep('report'); }
    catch (err) { setError(err.message); }
    finally { setReportPending(false); }
  }

  function newInterview() {
    window.speechSynthesis?.cancel();
    stopDevices();
    setStep('setup'); setConsent(false); setInterview(null); setStatus('created');
    setInterviewSetting('airport');
    setError(''); setInterimText(''); setTranscripts([]); setManualText('');
    setIntake({full_name:'',age:'',relationship_to_event:'',contact_reference:'',initial_statement:'',additional_notes:''});
    setQuestion(null); setReport(null); setReportPending(false); setFaceAnalysis(null);
    lastFaceStoreRef.current={pattern:'',time:0};
  }

  return <div className="app-shell">
    <header className="topbar">
      <div><div className="product-name">Immigration Interview Assistant</div><div className="subtitle">Airport and visa-office interview support</div></div>
      <div className="status-cluster"><Badge tone={listening?'red':'gray'}>{listening ? '● TRANSCRIBING' : 'TRANSCRIPTION OFF'}</Badge><Badge tone={camera || mic ? 'red' : 'gray'}>{camera || mic ? '● CAPTURE ACTIVE' : 'CAPTURE OFF'}</Badge></div>
    </header>
    <div className="disclaimer"><strong>Important:</strong> verify every transcript and generated question. This tool does not determine deception, admissibility, visa eligibility, or immigration status; an authorized officer retains all judgment.</div>
    {error && step !== 'report' && <div className="error" role="alert">{error}</div>}

    {step === 'setup' && <main className="setup-wrap">
      <div className="setup-copy"><Badge>Immigration interview setup</Badge><h1>Prepare a fair, neutral interview.</h1><p>Select whether the interview is taking place at an airport or visa office, choose the spoken language, and confirm informed consent. Audio is processed locally and deleted immediately.</p></div>
      <form className="setup-form" onSubmit={createInterview}>
        <label>Interview setting<select value={interviewSetting} onChange={event=>setInterviewSetting(event.target.value)}><option value="airport">Airport or border entry</option><option value="visa-office">Visa office or consulate</option></select></label>
        <label>Interview language<select value={language} onChange={event=>setLanguage(event.target.value)}>{languages.map(item=><option key={item.code} value={item.name}>{item.name}</option>)}</select></label>
        <fieldset className="intake-fields">
          <legend>Interviewee information</legend>
          <p>Record the applicant or traveller’s basic context and their initial travel or application summary before opening the live workspace.</p>
          <div className="intake-grid">
            <label>Full name<input required minLength="2" maxLength="160" value={intake.full_name} onChange={event=>setIntake(current=>({...current,full_name:event.target.value}))} autoComplete="name"/></label>
            <label>Age <span>(optional)</span><input type="number" min="1" max="130" value={intake.age} onChange={event=>setIntake(current=>({...current,age:event.target.value}))}/></label>
            <label>Travel or visa category<input required minLength="2" maxLength="200" value={intake.relationship_to_event} onChange={event=>setIntake(current=>({...current,relationship_to_event:event.target.value}))} placeholder={interviewSetting==='airport'?'For example: visitor, student, returning resident':'For example: tourist, student, work or family visa'}/></label>
            <label>Application or travel reference <span>(optional)</span><input maxLength="200" value={intake.contact_reference} onChange={event=>setIntake(current=>({...current,contact_reference:event.target.value}))} placeholder="Application, passport-safe, or case reference"/></label>
          </div>
          <label className="wide-field">Initial travel or application summary<textarea required minLength="10" maxLength="20000" rows="6" value={intake.initial_statement} onChange={event=>setIntake(current=>({...current,initial_statement:event.target.value}))} placeholder={interviewSetting==='airport'?'Record the traveller’s stated purpose, destination, intended stay, and relevant itinerary details.':'Record the applicant’s stated visa purpose, intended activities, travel plan, and relevant application details.'}/><small>This is treated as an earlier account, not as a verified fact.</small></label>
          <label className="wide-field">Additional intake notes <span>(optional)</span><textarea maxLength="5000" rows="3" value={intake.additional_notes} onChange={event=>setIntake(current=>({...current,additional_notes:event.target.value}))} placeholder="Relevant communication needs or administrative context only"/></label>
        </fieldset>
        <label className="check"><input type="checkbox" checked={consent} onChange={event=>setConsent(event.target.checked)}/><span>I confirm that informed consent has been obtained for device access and transcript storage.</span></label>
        <button className="primary" disabled={!consent || !intake.full_name.trim() || !intake.relationship_to_event.trim() || intake.initial_statement.trim().length<10}>Create interview and continue</button>
      </form>
    </main>}

    {step === 'live' && <main className="workspace">
      <aside className="control-rail">
        <div><span className="label">Interview</span><strong>#{interview.id} · {language}</strong></div>
        <div><span className="label">Setting</span><strong>{interviewSetting === 'airport' ? 'Airport / border entry' : 'Visa office / consulate'}</strong></div>
        <div><span className="label">Interviewee</span><strong>{intake.full_name}</strong></div>
        <div><span className="label">Status</span><Badge tone={status==='active'?'teal':'gray'}>{status}</Badge></div>
        <button onClick={checkDevices} disabled={checkingDevices || camera || mic}>{checkingDevices ? 'Checking devices…' : 'Check camera & mic'}</button>
        {(camera || mic) && <button onClick={stopDevices}>Stop camera & mic</button>}
        <button className="primary" onClick={()=>action('start')} disabled={status!=='created' || lifecyclePending}>{lifecyclePending ? 'Please wait…' : 'Start interview'}</button>
        <button onClick={()=>action(status==='paused'?'resume':'pause')} disabled={!['active','paused'].includes(status) || lifecyclePending}>{status==='paused'?'Resume':'Pause'}</button>
        <button onClick={listening ? stopListening : startListening} disabled={status!=='active' || !mic}>{listening ? 'Stop transcription' : 'Start transcription'}</button>
        <button onClick={generateQuestion} disabled={status!=='active' || (!transcripts.length && !manualText.trim()) || questionPending}>{questionPending ? 'Generating…' : 'Generate follow-up'}</button>
        <button className="danger" onClick={()=>action('end')} disabled={!['active','paused'].includes(status) || lifecyclePending}>End interview</button>
        <button onClick={makeReport} disabled={status!=='ended' || reportPending}>{reportPending ? 'Generating report\u2026' : 'Generate report'}</button>
        <button onClick={newInterview}>New interview</button>
      </aside>

      <div className="live-grid">
        <Section title="Live interview" className="video-panel">
          <div className="capture-indicator">{camera ? '● Camera active' : 'Camera inactive'} · {mic ? '● Microphone active' : 'Microphone inactive'}</div>
          <video ref={videoRef} autoPlay muted playsInline aria-label="Live camera preview"/>
          {!camera && <div className="video-empty">Allow camera access to display the live preview.</div>}
        </Section>

        <Section title="Current follow-up" className="question-panel">
          {question ? <p className="question">{question}</p> : <p className="empty">Start the interview to display the opening question.</p>}
          {question && <button onClick={()=>speakQuestion(question, listening)} disabled={status!=='active'}>Read question aloud</button>}
          <p className="reason">Questions are neutral and limited to relevant travel, application, document, itinerary, and clarification topics supported by the record.</p>
        </Section>

        <Section title="Live transcript" className="transcript-panel">
          <div className="translation-controls">
            <label className="check"><input type="checkbox" checked={translationEnabled} onChange={event=>setTranslationEnabled(event.target.checked)}/><span>Translate responses for the officer</span></label>
            <label>Officer’s reading language<select value={officerLanguage} onChange={event=>setOfficerLanguage(event.target.value)} disabled={!translationEnabled}>{languages.map(item=><option key={item.code} value={item.name}>{item.name}</option>)}</select></label>
            <p className="translation-note">Translation appears after each recorded segment is transcribed. Check important details with the speaker or an interpreter.</p>
          </div>
          <div className="transcript-list" aria-live="polite">
            {transcripts.map(item=><article key={item.id}><span>{new Date(item.created_at).toLocaleTimeString()} · Original ({item.language})</span><p dir="auto">{item.text}</p>{translationEnabled && <StatementTranslation interviewId={interview.id} statement={item} targetLanguage={officerLanguage}/>}</article>)}
            {interimText && <article className="interim"><span>Listening…</span><p>{interimText}</p></article>}
            {!transcripts.length && !interimText && <p className="empty">Start transcription and speak, or add a corrected statement below.</p>}
          </div>
          <div className="manual-entry">
            <label htmlFor="manual-transcript">Add or correct a statement</label>
            <textarea id="manual-transcript" value={manualText} onChange={event=>setManualText(event.target.value)} rows="3" placeholder="Enter the interviewee’s statement…" disabled={status!=='active'}/>
            <button onClick={saveManualTranscript} disabled={status!=='active' || !manualText.trim()}>Save statement</button>
          </div>
        </Section>

        <Section title="Facial activity (experimental)" className="face-panel">
          <div className="face-status"><Badge tone={faceAnalysisActive?'teal':'gray'}>{faceAnalysisLoading?'LOADING MODEL':faceAnalysisActive?'ANALYSIS ACTIVE':'ANALYSIS OFF'}</Badge></div>
          {faceAnalysis ? <div className="face-result"><strong>{faceAnalysis.pattern}</strong><div className="expression-estimate"><span>Estimated expression</span><b>{faceAnalysis.expression}</b><small>{Math.round(faceAnalysis.expressionConfidence*100)}% heuristic confidence</small></div><p>{faceAnalysis.interpretation}</p>{faceAnalysis.cues.length>0&&<ul>{faceAnalysis.cues.map(cue=><li key={cue.name}><span>{cue.name}</span><strong>{Math.round(cue.strength*100)}%</strong></li>)}</ul>}</div> : <p className="empty">Facial activity analysis starts with an active camera interview.</p>}
          <p className="reason">Expression estimates are computed locally from visible facial movements. They may not match the person’s actual emotion and cannot establish deception, intent, guilt, innocence, or truthfulness.</p>
        </Section>
      </div>
    </main>}

    {step === 'report' && <main className="report-wrap">
      <div className="report-head">
        <div><Badge>INTERVIEW REPORT</Badge><h1>{report.interview}</h1><p>Evidence-led review generated from the saved interview record</p></div>
        <button onClick={()=>window.print()}>Print / Save PDF</button>
      </div>

      <section className="report-meta" aria-label="Report details">
        <dl><div><dt>Interview ID</dt><dd>#{report.metadata.interview_id}</dd></div><div><dt>Language</dt><dd>{report.metadata.language}</dd></div><div><dt>Duration</dt><dd>{formatDuration(report.analytics.duration_seconds)}</dd></div><div><dt>Generated</dt><dd>{formatDate(report.metadata.generated_at)}</dd></div></dl>
        <div className="report-objective"><span>Objective</span><p>{report.metadata.objective}</p></div>
      </section>

      {report.metadata.interviewee && <section className="interviewee-summary" aria-labelledby="interviewee-summary-title">
        <div><h2 id="interviewee-summary-title">Applicant or traveller intake</h2><dl><div><dt>Full name</dt><dd>{report.metadata.interviewee.full_name}</dd></div><div><dt>Age</dt><dd>{report.metadata.interviewee.age || 'Not provided'}</dd></div><div><dt>Travel or visa category</dt><dd>{report.metadata.interviewee.relationship_to_event}</dd></div><div><dt>Application/travel reference</dt><dd>{report.metadata.interviewee.contact_reference || 'Not provided'}</dd></div></dl></div>
        <div className="prior-statement"><span>Initial travel or application summary</span><blockquote>{report.metadata.interviewee.initial_statement}</blockquote>{report.metadata.interviewee.additional_notes && <p><strong>Additional notes:</strong> {report.metadata.interviewee.additional_notes}</p>}</div>
      </section>}

      <section className="report-metrics" aria-label="Interview record summary">
        <div><strong>{report.analytics.statement_count}</strong><span>Saved statements</span></div>
        <div><strong>{report.analytics.question_count}</strong><span>Questions asked</span></div>
        <div><strong>{report.analytics.total_words}</strong><span>Transcript words</span></div>
        <div><strong>{report.analytics.observation_count}</strong><span>Facial observations</span></div>
        <div><strong>{report.analytics.association_count}</strong><span>Statement associations</span></div>
        <div><strong>{Math.round(report.analytics.average_transcript_confidence*100)}%</strong><span>Average STT confidence</span></div>
      </section>

      <div className="report-narrative">
        <Section title="Recorded account"><ReportList items={report.observed}/></Section>
        <Section title="Analytical notes"><ReportList items={report.interpreted}/></Section>
        <Section title="Recommended next steps"><ReportList items={report.recommended}/></Section>
        <Section title="Evidence gaps and limitations"><ReportList items={report.limitations}/></Section>
      </div>

      <section className="report-visuals" aria-label="Calculated interview charts">
        <Section title="Statement detail"><BarChart data={report.analytics.statement_word_counts} valueLabel=" words"/></Section>
        <Section title="Observed facial movement patterns"><BarChart data={report.analytics.facial_pattern_counts} valueLabel=""/></Section>
        <Section title="Estimated expression distribution"><BarChart data={report.analytics.expression_estimate_counts || []} valueLabel=""/><p className="chart-note">Counts are uncertain camera-based expression estimates, not verified emotions.</p></Section>
        <Section title="Facial movement detection strength" className="strength-panel"><StrengthChart data={report.analytics.facial_strength_timeline}/><p className="chart-note">Values show MediaPipe detection strength, not emotional intensity or truthfulness.</p></Section>
      </section>

      <Section title="Statement and facial activity associations" className="association-section">
        {report.statement_facial_associations.length ? <div className="association-list">{report.statement_facial_associations.map((item,index)=><article key={`${item.observed_at}-${index}`}>
          <div><span>Statement</span><p>“{item.statement.text}”</p><small>{formatDate(item.statement.created_at)}</small></div>
          <div><span>Recorded nearby</span><p>{item.facial_pattern || 'Unclassified facial movement'} · {item.expression_estimate || 'Expression undetermined'} · {Math.round(item.strength*100)}% detection strength</p><small>{formatDate(item.observed_at)}</small></div>
          <p className="association-caution">Temporal co-occurrence only. The estimated expression may not represent the person’s actual emotion and is not evidence of truthfulness.</p>
        </article>)}</div> : <p className="empty">No facial observations were associated with a saved statement.</p>}
      </Section>

      <Section title="Transcript record" className="appendix-section">
        <div className="report-transcript">{report.transcript.map((item,index)=><article key={item.id}><header><strong>Statement {index+1}</strong><span>{formatDate(item.created_at)} · {Math.round(item.confidence*100)}% recognition confidence</span></header><p>{item.text}</p></article>)}</div>
      </Section>

      <Section title="Question record" className="appendix-section">
        <ol className="question-record">{report.questions.map(item=><li key={item.id}><strong>{item.question}</strong><span>{item.strategy}</span><p>{item.reason}</p></li>)}</ol>
      </Section>

      <div className="method-note"><strong>Review boundary</strong><p>This report combines saved transcript text, generated questions, and recorded facial movement measurements. It does not determine deception, admissibility, visa eligibility, immigration status, intent, personality, or mental state. Verify the record and apply the governing immigration procedure independently.</p><span>Model: {report.model} · Raw media stored: {report.metadata.raw_media_stored ? 'Yes' : 'No'}</span></div>
      <div className="report-actions"><button onClick={()=>setStep('live')}>Back to interview</button><button onClick={newInterview}>Start new interview</button></div>
    </main>}
    <footer>Student project · Human review required · No automated immigration or deception decisions</footer>
  </div>;
}
