// Metro configuration
// 추가로 tflite 모델 파일을 에셋으로 번들하기 위해 확장자를 등록한다.
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// tflite를 자바스크립트 소스가 아닌 에셋으로 처리
config.resolver.assetExts.push("tflite");
config.resolver.sourceExts = config.resolver.sourceExts.filter((ext) => ext !== "tflite");

module.exports = config;
