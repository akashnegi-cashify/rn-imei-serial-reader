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
import { useTextRecognition } from 'react-native-vision-camera-text-recognition';
import { CAPTURE_MODE } from '../captureMode';
import { createParser } from '../parsers';
import { toRecognizedText } from '../adapters/mlkitAdapter';
import { nativeFrameToJpeg } from '../adapters/nativeFrameToJpeg';
import type { Frame, FrameOrientation, ParserConfig } from '../types';

const GRACE_MS = 1000;
const TARGET_FPS = 10;
const TEXT_RECOGNITION_OPTIONS = { language: 'latin' } as const;
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
  // Cap photo resolution. Only relevant when CAPTURE_MODE === 'take-photo'
  // (default formats are 12+MP which makes takePhoto encode slowly). Harmless
  // in 'native-frame' mode where takePhoto isn't called.
  const format = useCameraFormat(device, [
    { photoResolution: { width: 1920, height: 1080 } },
    { videoResolution: { width: 1280, height: 720 } },
  ]);

  const isBusy = useSharedValue<boolean>(false);
  const graceUntil = useSharedValue<number>(0);

  const onDoneRef = useRef(opts.onDone);
  onDoneRef.current = opts.onDone;
  const onErrorRef = useRef(opts.onError);
  onErrorRef.current = opts.onError;
  // Used by the takePhoto path (decision made on JS thread).
  const captureFrameRef = useRef(!!opts.captureFrame);
  captureFrameRef.current = !!opts.captureFrame;

  // Used by the native-frame path (decision made inside the worklet, since
  // nativeFrameToJpeg has to run while the Frame is alive).
  const captureFrameFlag = useSharedValue<boolean>(!!opts.captureFrame);
  useEffect(() => {
    captureFrameFlag.value = !!opts.captureFrame;
  }, [opts.captureFrame, captureFrameFlag]);

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

  const handleMatch = useCallback(
    async (
      values: string[],
      path: string | null,
      width: number,
      height: number,
      orientation: FrameOrientation | null,
    ) => {
      try {
        let frame: Frame | undefined;
        if (CAPTURE_MODE === 'native-frame') {
          // Worklet already wrote the JPEG and gave us its path. No async work.
          if (path != null && orientation != null) {
            frame = { uri: `file://${path}`, width, height, orientation };
          }
        } else {
          // CAPTURE_MODE === 'take-photo'
          if (captureFrameRef.current && cameraRef.current != null) {
            const photo = await cameraRef.current.takePhoto({
              enableShutterSound: false,
              flash: 'off',
            });
            frame = {
              uri: `file://${photo.path}`,
              width: photo.width,
              height: photo.height,
              orientation: photo.orientation as FrameOrientation,
            };
          }
        }
        onDoneRef.current(values, frame);
      } catch (e) {
        const err = e as Error;
        setError(err);
        onErrorRef.current?.(err);
      } finally {
        isBusy.value = false;
      }
    },
    [isBusy],
  );

  const reportError = useCallback((e: Error) => {
    setError(e);
    onErrorRef.current?.(e);
  }, []);

  const handleMatchJs = useMemo(() => Worklets.createRunOnJS(handleMatch), [handleMatch]);
  const reportErrorJs = useMemo(() => Worklets.createRunOnJS(reportError), [reportError]);

  const { scanText } = useTextRecognition(TEXT_RECOGNITION_OPTIONS);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (parser == null) return;
      runAtTargetFps(TARGET_FPS, () => {
        'worklet';
        if (isBusy.value) return;
        if (Date.now() < graceUntil.value) return;

        // Run scanText synchronously on the camera frame thread. The
        // original code used `runAsync(frame, ...)` to push OCR onto a
        // second worklet runtime, but under React Native's new architecture
        // (bridgeless mode) that secondary runtime crashes with SIGSEGV in
        // `JsiWorkletContext::invokeOnWorkletThread` on Android — confirmed
        // tombstone with backtrace going through librnworklets.so →
        // libVisionCamera.so → CameraView.onFrame. We're already throttled
        // to TARGET_FPS via runAtTargetFps, so synchronous OCR is fine.
        try {
          const raw = scanText(frame);
          const rt = toRecognizedText(raw);
          const values = parser(rt);
          if (values != null && values.length > 0) {
            isBusy.value = true;
            if (CAPTURE_MODE === 'native-frame' && captureFrameFlag.value) {
              const ext = nativeFrameToJpeg(frame, JPEG_QUALITY);
              handleMatchJs(values, ext.path, ext.width, ext.height, ext.orientation);
            } else {
              handleMatchJs(values, null, 0, 0, null);
            }
          }
        } catch (e) {
          reportErrorJs(e as Error);
        }
      });
    },
    [parser, scanText, handleMatchJs, reportErrorJs, isBusy, graceUntil, captureFrameFlag],
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
