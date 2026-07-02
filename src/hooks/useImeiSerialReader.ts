import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  type CameraDevice,
  type CameraDeviceFormat,
  runAtTargetFps,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets, useSharedValue } from 'react-native-worklets-core';
import { PhotoRecognizer } from 'react-native-vision-camera-text-recognition';
import { createParser } from '../parsers';
import { toRecognizedText } from '../adapters/mlkitAdapter';
import { nativeFrameToJpeg } from '../adapters/nativeFrameToJpeg';
import type { Frame, FrameOrientation, ParserConfig } from '../types';

const GRACE_MS = 1000;
const TARGET_FPS = 10;
const JPEG_QUALITY = 80;

export interface UseImeiSerialReaderOptions {
  parserConfig: ParserConfig;
  onDone: (values: string[], frame?: Frame) => void;
  onError?: (error: Error) => void;
  captureFrame?: boolean;
}

export interface UseImeiSerialReaderReturn {
  cameraRef: React.RefObject<Camera | null>;
  isActive: boolean;
  reload: () => void;
  error: Error | null;
  device: CameraDevice | undefined;
  format: CameraDeviceFormat | undefined;
  hasPermission: boolean;
  requestPermission: () => Promise<boolean>;
  frameProcessor: ReturnType<typeof useFrameProcessor>;
}

export function useImeiSerialReader(opts: UseImeiSerialReaderOptions): UseImeiSerialReaderReturn {
  const cameraRef = useRef<Camera>(null);
  const [isActive, setIsActive] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  // Cap video/photo resolution. 1280x720 keeps the frames light enough for
  // fast YUV→JPEG conversion and ML Kit text recognition.
  const format = useCameraFormat(device, [
    { photoResolution: { width: 1920, height: 1080 } },
    { videoResolution: { width: 1280, height: 720 } },
  ]);

  const isBusy = useSharedValue<boolean>(false);
  const graceUntil = useSharedValue<number>(0);
  // True while a JS-thread PhotoRecognizer call is in flight. Prevents the
  // frame processor from queueing up multiple OCR requests — one attempt
  // completes before the next frame is even considered.
  const isProcessing = useSharedValue<boolean>(false);

  const onDoneRef = useRef(opts.onDone);
  onDoneRef.current = opts.onDone;
  const onErrorRef = useRef(opts.onError);
  onErrorRef.current = opts.onError;
  // OCR runs on the JS thread now, so this ref is read from `recognizeAndMatch`.
  const captureFrameRef = useRef(!!opts.captureFrame);
  captureFrameRef.current = !!opts.captureFrame;

  const parserResult = useMemo<{ parser: ReturnType<typeof createParser> | null; error: Error | null }>(() => {
    try {
      return { parser: createParser(opts.parserConfig), error: null };
    } catch (e) {
      return { parser: null, error: e as Error };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    opts.parserConfig.readerType,
    opts.parserConfig.customRegex,
    opts.parserConfig.targetBarcode,
    opts.parserConfig.minLength,
    opts.parserConfig.maxLength,
  ]);
  const parser = parserResult.parser;

  useEffect(() => {
    setError(parserResult.error);
    if (parserResult.error != null) {
      onErrorRef.current?.(parserResult.error);
    }
  }, [parserResult]);

  useEffect(() => {
    if (isActive) {
      isBusy.value = false;
      graceUntil.value = Date.now() + GRACE_MS;
    }
  }, [isActive, isBusy, graceUntil]);

  const reportError = useCallback((e: Error) => {
    setError(e);
    onErrorRef.current?.(e);
  }, []);
  const reportErrorJs = useMemo(() => Worklets.createRunOnJS(reportError), [reportError]);

  // JS-thread OCR + match handler. Called via `runOnJS` from the frame
  // processor with a JPEG file path (already written on the frame thread).
  // PhotoRecognizer is a plain NativeModule call — no worklets involved —
  // so it avoids the SIGSEGV in worklets-core's `invokeOnWorkletThread`
  // that killed the old `runAsync(frame, ...)` path under bridgeless mode.
  const recognizeAndMatch = useCallback(
    async (path: string, width: number, height: number, orientation: FrameOrientation) => {
      try {
        if (parser == null) return;
        const raw = await PhotoRecognizer({ uri: `file://${path}`, orientation });
        const rt = toRecognizedText(raw);
        const values = parser(rt);
        if (values != null && values.length > 0) {
          isBusy.value = true;
          const frameForConsumer: Frame | undefined = captureFrameRef.current
            ? { uri: `file://${path}`, width, height, orientation }
            : undefined;
          onDoneRef.current(values, frameForConsumer);
        }
      } catch (e) {
        const err = e as Error;
        setError(err);
        onErrorRef.current?.(err);
      } finally {
        isProcessing.value = false;
      }
    },
    [parser, isBusy, isProcessing],
  );
  const recognizeAndMatchJs = useMemo(
    () => Worklets.createRunOnJS(recognizeAndMatch),
    [recognizeAndMatch],
  );

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (parser == null) return;
      runAtTargetFps(TARGET_FPS, () => {
        'worklet';
        if (isBusy.value) return;
        if (isProcessing.value) return;
        if (Date.now() < graceUntil.value) return;

        // Do only the fast, synchronous YUV→JPEG conversion on the camera
        // frame thread (~20-50ms), then hand the path to the JS thread for
        // OCR via PhotoRecognizer. This keeps the camera pipeline
        // unblocked — preview stays smooth even though ML Kit runs off
        // the frame thread. Also sidesteps the bridgeless SIGSEGV
        // triggered by vision-camera's `runAsync` + worklets-core
        // secondary runtime path.
        try {
          isProcessing.value = true;
          const ext = nativeFrameToJpeg(frame, JPEG_QUALITY);
          recognizeAndMatchJs(ext.path, ext.width, ext.height, ext.orientation);
        } catch (e) {
          isProcessing.value = false;
          reportErrorJs(e as Error);
        }
      });
    },
    [parser, recognizeAndMatchJs, reportErrorJs, isBusy, isProcessing, graceUntil],
  );

  const reload = useCallback(() => {
    setError(null);
    setIsActive(false);
    setTimeout(() => setIsActive(true), 50);
  }, []);

  useEffect(() => {
    if (!hasPermission) {
      void requestPermission();
    }
  }, [hasPermission, requestPermission]);

  return {
    cameraRef,
    isActive,
    reload,
    error,
    device,
    format,
    hasPermission,
    requestPermission,
    frameProcessor,
  };
}
