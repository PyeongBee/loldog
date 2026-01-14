// Metro configuration
// 추가로 onnx 모델 파일을 에셋으로 번들하기 위해 확장자를 등록한다.
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// onnx를 자바스크립트 소스가 아닌 에셋으로 처리
config.resolver.assetExts.push("onnx");
config.resolver.sourceExts = config.resolver.sourceExts.filter((ext) => ext !== "onnx");

module.exports = config;
