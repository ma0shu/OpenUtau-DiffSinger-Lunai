using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Api;
using OpenUtau.Core;
using OpenUtau.Core.Format;
using OpenUtau.Core.Util;
using OpenUtau.Core.Ustx;
using Serilog;

namespace OpenUtau.Core.Analysis;



public sealed class HubertFAOptions {
    public string? ModelPath { get; set; }
    public string G2p { get; set; } = "dictionary";
    public string NonLexicalPhonemes { get; set; } = "AP";
    public string Language { get; set; } = "zh";
    public string? DictionaryPath { get; set; }
    public IG2p? G2pProvider { get; set; }
    public int PadTimes { get; set; } = 1;
    public int PadLength { get; set; } = 5;
}

public sealed class HubertFAWordTiming {
    public double StartSec { get; init; }
    public double EndSec { get; init; }
    public string Text { get; init; } = string.Empty;
}

public static class HubertFA {
    public const string PackageId = "hubertfa";
    private const string DefaultModelRelativePath = "models\\1218_hfa_model_new_dict\\model.onnx";

    public static bool IsInstalled(string? modelPath = null) {
        return TryResolveModelPath(out _, modelPath);
    }

    public static bool TryResolveModelPath(out string path, string? modelPath = null) {
        try {
            path = ResolveModelPath(modelPath);
            return true;
        } catch {
            path = string.Empty;
            return false;
        }
    }

    public static string ResolveModelPath(string? modelPath = null) {
        if (!string.IsNullOrWhiteSpace(modelPath)) {
            var fullPath = Path.GetFullPath(modelPath);
            if (!File.Exists(fullPath)) {
                throw new FileNotFoundException("HubertFA model not found.", fullPath);
            }
            return fullPath;
        }

        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void AddCandidateBase(string? baseDir) {
            if (string.IsNullOrWhiteSpace(baseDir)) {
                return;
            }
            candidates.Add(Path.GetFullPath(Path.Combine(baseDir, "HUBERTFA-REWRITE", DefaultModelRelativePath)));
            candidates.Add(Path.GetFullPath(Path.Combine(baseDir, DefaultModelRelativePath)));
        }

        AddCandidateBase(PathManager.Inst.RootPath);
        AddCandidateBase(AppContext.BaseDirectory);
        AddCandidateBase(Environment.CurrentDirectory);

        var installed = PackageManager.Inst.GetInstalledPath(PackageId);
        if (!string.IsNullOrWhiteSpace(installed)) {
            AddCandidateBase(installed);
            candidates.Add(Path.GetFullPath(Path.Combine(installed, "model.onnx")));
        }

        foreach (var baseDir in EnumerateParents(PathManager.Inst.RootPath, 8)) {
            AddCandidateBase(baseDir);
        }
        foreach (var baseDir in EnumerateParents(AppContext.BaseDirectory, 8)) {
            AddCandidateBase(baseDir);
        }
        foreach (var baseDir in EnumerateParents(Environment.CurrentDirectory, 8)) {
            AddCandidateBase(baseDir);
        }

        foreach (var candidate in candidates) {
            if (File.Exists(candidate)) {
                return candidate;
            }
        }

        throw new FileNotFoundException(
            "HubertFA model not found. Expected model.onnx in HUBERTFA-REWRITE/models/1218_hfa_model_new_dict.",
            candidates.FirstOrDefault() ?? DefaultModelRelativePath);
    }

    public static IReadOnlyList<HubertFAWordTiming> InferWordTimings(
        string wavPath,
        string lyrics,
        HubertFAOptions? options = null,
        Action<string>? progress = null) {

        if (string.IsNullOrWhiteSpace(wavPath)) {
            throw new ArgumentException("WAV path is empty.", nameof(wavPath));
        }
        if (!File.Exists(wavPath)) {
            throw new FileNotFoundException("WAV file not found.", wavPath);
        }
        if (string.IsNullOrWhiteSpace(lyrics)) {
            throw new ArgumentException("Lyrics is empty.", nameof(lyrics));
        }

        options ??= new HubertFAOptions();
        if (options.PadTimes < 1) {
            throw new ArgumentException("PadTimes must be >= 1.", nameof(options));
        }
        if (options.PadLength < 0) {
            throw new ArgumentException("PadLength must be >= 0.", nameof(options));
        }

        var modelPath = ResolveModelPath(options.ModelPath);

        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

        progress?.Invoke("HubertFA 15%: Loading options");
        progress?.Invoke("HubertFA 20%: Loading config");
        using var runner = new InferenceOnnxRunner(modelPath, options);
        runner.LoadConfig();

        progress?.Invoke("HubertFA 25%: Initializing decoder");
        runner.InitDecoder();

        progress?.Invoke("HubertFA 30%: Loading model");
        runner.LoadModel();

        progress?.Invoke("HubertFA 35%: Running inference, please wait...");
        runner.AddSample(wavPath, lyrics);
        runner.Infer();

        progress?.Invoke("HubertFA 90%: Collecting word timings");
        if (runner.Predictions.Count == 0) {
            return Array.Empty<HubertFAWordTiming>();
        }
        var words = runner.Predictions[0].Words;
        if (words.Count == 0) {
            return Array.Empty<HubertFAWordTiming>();
        }
        return words
            .Select(word => new HubertFAWordTiming {
                StartSec = word.Start,
                EndSec = word.End,
                Text = word.Text,
            })
            .ToList();
    }

    private static IEnumerable<string> EnumerateParents(string? startPath, int maxDepth) {
        if (string.IsNullOrWhiteSpace(startPath)) {
            yield break;
        }

        var dir = new DirectoryInfo(Path.GetFullPath(startPath));
        if (!dir.Exists && dir.Parent != null) {
            dir = dir.Parent;
        }

        var depth = 0;
        while (dir != null && depth < maxDepth) {
            yield return dir.FullName;
            dir = dir.Parent;
            depth++;
        }
    }
}



internal sealed class InferenceOnnxRunner : IDisposable {
    private readonly HubertFAOptions options;
    private readonly List<DatasetItem> dataset = new();
    private readonly List<PredictionItem> predictions = new();

    private readonly string modelPath;
    private readonly string modelFolder;

    private VocabConfig? vocab;
    private MelSpecConfig? melCfg;
    private AlignmentDecoder? faDecoder;
    private NonLexicalDecoder? nllDecoder;
    private IG2p? dictionaryG2p;
    private InferenceSession? model;

    public IReadOnlyList<PredictionItem> Predictions => predictions;

    public InferenceOnnxRunner(string modelPath, HubertFAOptions options) {
        this.options = options;
        this.modelPath = Path.GetFullPath(modelPath);
        modelFolder = Path.GetDirectoryName(this.modelPath) ?? throw new InvalidOperationException("Invalid model path.");
    }

    public void LoadConfig() {
        EnsureConfigFile("vocab.json");
        EnsureConfigFile("config.json");
        EnsureConfigFile("VERSION");

        var versionText = File.ReadAllText(Path.Combine(modelFolder, "VERSION")).Trim();
        if (!int.TryParse(versionText, out var version) || version != 5) {
            throw new InvalidOperationException($"ONNX model version must be 5. got '{versionText}'");
        }

        var jsonOptions = new JsonSerializerOptions {
            PropertyNameCaseInsensitive = true,
        };

        vocab = JsonSerializer.Deserialize<VocabConfig>(File.ReadAllText(Path.Combine(modelFolder, "vocab.json")), jsonOptions)
            ?? throw new InvalidOperationException("Failed to parse vocab.json");

        var config = JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(Path.Combine(modelFolder, "config.json")), jsonOptions)
            ?? throw new InvalidOperationException("Failed to parse config.json");

        melCfg = config.MelSpecConfig;

        foreach (var (_, dictFile) in vocab.Dictionaries) {
            var dictPath = Path.Combine(modelFolder, dictFile);
            if (!File.Exists(dictPath)) {
                throw new FileNotFoundException($"Dictionary not found: {dictPath}");
            }
        }
    }

    public void InitDecoder() {
        EnsureConfigsLoaded();
        faDecoder = new AlignmentDecoder(vocab!, melCfg!.SampleRate, melCfg.HopSize);
        nllDecoder = new NonLexicalDecoder(new List<string> { "None" }.Concat(vocab!.NonLexicalPhonemes).ToList(), melCfg.SampleRate, melCfg.HopSize);
    }

    public void LoadModel() {
        model = Onnx.getInferenceSession(modelPath, OnnxRunnerChoice.CPU);
    }

    public void AddSample(string wavPath, string lyrics) {
        EnsureConfigsLoaded();
        if (string.IsNullOrWhiteSpace(wavPath)) {
            throw new ArgumentException("WAV path is empty.", nameof(wavPath));
        }
        if (!File.Exists(wavPath)) {
            throw new FileNotFoundException("WAV file not found.", wavPath);
        }

        var labText = lyrics?.Trim() ?? string.Empty;
        (List<string> PhSeq, List<string> WordSeq, List<int> PhIdxToWordIdx) converted = options.G2p.ToLowerInvariant() switch {
            "dictionary" => ConvertLyricsByDictionary(labText),
            "phoneme" => ConvertLyricsByPhoneme(labText),
            _ => throw new ArgumentException("g2p must be dictionary or phoneme"),
        };
        dataset.Add(new DatasetItem(wavPath, converted.PhSeq, converted.WordSeq, converted.PhIdxToWordIdx));
    }

    private (List<string> PhSeq, List<string> WordSeq, List<int> PhIdxToWordIdx) ConvertLyricsByPhoneme(string text) {
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Where(ph => ph != "SP")
            .ToList();
        var phSeq = new List<string> { "SP" };
        var mapping = new List<int> { -1 };
        for (var i = 0; i < words.Count; i++) {
            phSeq.Add(words[i]);
            mapping.Add(i);
            phSeq.Add("SP");
            mapping.Add(-1);
        }
        return FinalizeConvertedPhonemes(phSeq, words, mapping);
    }

    private (List<string> PhSeq, List<string> WordSeq, List<int> PhIdxToWordIdx) ConvertLyricsByDictionary(string text) {
        var g2p = GetDictionaryG2p();
        var wordSeq = new List<string>();
        var phSeq = new List<string> { "SP" };
        var mapping = new List<int> { -1 };
        foreach (var rawWord in text.Split(' ', StringSplitOptions.RemoveEmptyEntries)) {
            var word = rawWord.Trim();
            if (word.Length == 0) {
                continue;
            }
            var phones = g2p.Query(word) ?? g2p.Query(word.ToLowerInvariant());
            if (phones == null || phones.Length == 0) {
                Log.Warning("HubertFA: word {Word} not in dictionary, skipped.", word);
                continue;
            }
            wordSeq.Add(word);
            var wordSeqIdx = wordSeq.Count - 1;
            for (var i = 0; i < phones.Length; i++) {
                var ph = phones[i];
                if (string.IsNullOrWhiteSpace(ph)) {
                    continue;
                }
                if ((i == 0 || i == phones.Length - 1) && ph == "SP") {
                    Log.Warning("HubertFA: dictionary word {Word} starts/ends with SP, ignored.", word);
                    continue;
                }
                phSeq.Add(ph);
                mapping.Add(wordSeqIdx);
            }
            if (phSeq[^1] != "SP") {
                phSeq.Add("SP");
                mapping.Add(-1);
            }
        }
        return FinalizeConvertedPhonemes(phSeq, wordSeq, mapping);
    }

    private string ResolveDictionaryPath() {
        EnsureConfigsLoaded();
        if (!string.IsNullOrWhiteSpace(options.DictionaryPath)) {
            return Path.GetFullPath(options.DictionaryPath);
        }
        if (!vocab!.Dictionaries.TryGetValue(options.Language, out var dictName)) {
            throw new InvalidOperationException($"Language '{options.Language}' not found in vocab dictionaries.");
        }
        return Path.Combine(modelFolder, dictName);
    }
    private IG2p GetDictionaryG2p() {
        EnsureConfigsLoaded();
        if (dictionaryG2p != null) {
            return dictionaryG2p;
        }
        if (options.G2pProvider != null) {
            dictionaryG2p = options.G2pProvider;
            return dictionaryG2p;
        }
        var dictPath = ResolveDictionaryPath();
        
        var builder = G2pDictionary.NewBuilder();
        if (dictPath.EndsWith(".yaml", StringComparison.OrdinalIgnoreCase) ||
            dictPath.EndsWith(".yml", StringComparison.OrdinalIgnoreCase)) {
            var text = File.ReadAllText(dictPath);
            builder.Load(text);
            dictionaryG2p = builder.Build();
            return dictionaryG2p;
        } 
        
        // Load text dictionary
        builder.AddSymbol("SP", isVowel: false);
        foreach (var line in File.ReadLines(dictPath)) {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var trimmed = line.Trim();
            if (trimmed.StartsWith("#", StringComparison.Ordinal)) continue;
            
            var parts = trimmed.Split('\t', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            
            var word = parts[0].Trim();
            var phones = parts[1].Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(ph => ph.Trim())
                .Where(ph => ph.Length > 0)
                .ToArray();
                
            if (word.Length == 0 || phones.Length == 0) continue;
            
            foreach(var ph in phones) {
                builder.AddSymbol(ph, isVowel: false);
            }
            builder.AddEntry(word, phones);
        }
        
        dictionaryG2p = builder.Build();
        return dictionaryG2p;
    }

    private (List<string> PhSeq, List<string> WordSeq, List<int> PhIdxToWordIdx) FinalizeConvertedPhonemes(
        List<string> phSeq,
        List<string> wordSeq,
        List<int> phIdxToWordIdx) {
        if (phSeq.Count == 0 || phSeq[0] != "SP" || phSeq[^1] != "SP") {
            throw new InvalidOperationException("Phoneme sequence must start and end with SP.");
        }
        for (var i = 0; i < phSeq.Count - 1; i++) {
            if (phSeq[i] == "SP" && phSeq[i + 1] == "SP") {
                throw new InvalidOperationException("Consecutive SP phonemes are not allowed.");
            }
        }
        var language = vocab!.LanguagePrefix ? options.Language : null;
        if (!string.IsNullOrEmpty(language)) {
            for (var i = 0; i < phSeq.Count; i++) {
                if (phSeq[i] != "SP") {
                    phSeq[i] = $"{language}/{phSeq[i]}";
                }
            }
        }
        return (phSeq, wordSeq, phIdxToWordIdx);
    }

    public void Infer() {
        EnsureReadyForInfer();
        predictions.Clear();
        if (dataset.Count == 0) {
            return;
        }

        var nonLexicalPhonemes = options.NonLexicalPhonemes
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(x => x.Trim())
            .Where(x => x.Length > 0)
            .ToList();

        var vocabSet = vocab!.NonLexicalPhonemes.ToHashSet();
        if (nonLexicalPhonemes.Any(ph => !vocabSet.Contains(ph))) {
            var invalid = string.Join(", ", nonLexicalPhonemes.Where(ph => !vocabSet.Contains(ph)));
            throw new InvalidOperationException($"Invalid non lexical phonemes: {invalid}");
        }

        var padLengths = options.PadTimes > 1
            ? Enumerable.Range(0, options.PadTimes)
                .Select(i => Math.Round(options.PadLength / (double)options.PadTimes * i, 1))
                .ToList()
            : new List<double> { 0 };

        for (var sampleIdx = 0; sampleIdx < dataset.Count; sampleIdx++) {
            var sample = dataset[sampleIdx];
            Log.Information("HubertFA: infer {Index}/{Count} {WavPath}", sampleIdx + 1, dataset.Count, sample.WavPath);

            using var waveStream = Wave.OpenFile(sample.WavPath);
            var wav = Wave.GetMonoSamples(waveStream, melCfg!.SampleRate);
            var wavLength = wav.Length / (double)melCfg.SampleRate;

            var wordsList = new List<WordList>();
            foreach (var padLen in padLengths) {
                var paddedSamples = (int)(padLen * melCfg.SampleRate);
                var paddedFrames = (int)(paddedSamples / (double)melCfg.HopSize);

                var paddedWav = new float[paddedSamples + wav.Length];
                Array.Copy(wav, 0, paddedWav, paddedSamples, wav.Length);

                var (words, nonLexicalWords) = InferSingle(
                    paddedWav,
                    paddedFrames,
                    sample.WordSeq,
                    sample.PhSeq,
                    sample.PhIdxToWordIdx,
                    wavLength,
                    nonLexicalPhonemes
                );

                foreach (var tagWords in nonLexicalWords) {
                    foreach (var w in tagWords) {
                        words.AddAp(w);
                    }
                }

                words.ClearLanguagePrefix();
                wordsList.Add(words);
            }

            List<WordList> selected;
            try {
                var bestIndices = InferMath.FindAllDuplicatePhonemes(wordsList.Select(w => w.Phonemes).ToList());
                selected = bestIndices.Select(i => wordsList[i]).ToList();
            } catch (Exception ex) {
                Log.Warning(ex, "HubertFA: duplicate pad group not found, fallback to first result.");
                selected = new List<WordList> { wordsList[0] };
            }

            var resultWord = MergeWordLists(selected, wavLength);
            resultWord.FillSmallGaps(wavLength);
            resultWord.AddSp(wavLength);

            var warningLog = resultWord.Log();
            if (!string.IsNullOrWhiteSpace(warningLog)) {
                Log.Warning("HubertFA: {Warning}", warningLog);
            }

            predictions.Add(new PredictionItem(resultWord));
        }
    }

    private (WordList Words, List<List<Word>> NonLexicalWords) InferSingle(
        float[] paddedWav,
        int paddedFrames,
        List<string> wordSeq,
        List<string> phSeq,
        List<int> phIdxToWordIdx,
        double wavLength,
        List<string> nonLexicalPhonemes
    ) {
        var waveformTensor = new DenseTensor<float>(new[] { 1, paddedWav.Length });
        for (var i = 0; i < paddedWav.Length; i++) {
            waveformTensor[0, i] = paddedWav[i];
        }

        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("waveform", waveformTensor),
        };

        using var results = model!.Run(inputs, model.OutputNames);

        float[,] phFrameLogits;
        float[] phEdgeLogits;
        float[,] cvntLogits;

        var outputMap = results.ToDictionary(x => x.Name, x => x.AsTensor<float>());
        phFrameLogits = Extract2D(outputMap["ph_frame_logits"], paddedFrames);
        phEdgeLogits = Extract1D(outputMap["ph_edge_logits"], paddedFrames);
        cvntLogits = Extract2D(outputMap["cvnt_logits"], paddedFrames);

        var (words, _) = faDecoder!.Decode(
            phFrameLogits,
            phEdgeLogits,
            wavLength,
            phSeq,
            wordSeq,
            phIdxToWordIdx
        );

        var nonLexicalWords = nllDecoder!.Decode(
            cvntLogits,
            wavLength,
            nonLexicalPhonemes
        );

        return (words, nonLexicalWords);
    }

    private static float[,] Extract2D(Tensor<float> tensor, int startFrame) {
        if (tensor.Dimensions.Length == 3) {
            var c = tensor.Dimensions[1];
            var t = tensor.Dimensions[2];
            var tOut = Math.Max(0, t - startFrame);
            var output = new float[c, tOut];
            for (var ci = 0; ci < c; ci++) {
                for (var ti = 0; ti < tOut; ti++) {
                    output[ci, ti] = tensor[0, ci, ti + startFrame];
                }
            }
            return output;
        }

        if (tensor.Dimensions.Length == 2) {
            var c = tensor.Dimensions[0];
            var t = tensor.Dimensions[1];
            var tOut = Math.Max(0, t - startFrame);
            var output = new float[c, tOut];
            for (var ci = 0; ci < c; ci++) {
                for (var ti = 0; ti < tOut; ti++) {
                    output[ci, ti] = tensor[ci, ti + startFrame];
                }
            }
            return output;
        }

        throw new InvalidOperationException($"Unexpected tensor rank for 2D extraction: {tensor.Dimensions.Length}");
    }

    private static float[] Extract1D(Tensor<float> tensor, int startFrame) {
        if (tensor.Dimensions.Length == 2) {
            var t = tensor.Dimensions[1];
            var tOut = Math.Max(0, t - startFrame);
            var output = new float[tOut];
            for (var ti = 0; ti < tOut; ti++) {
                output[ti] = tensor[0, ti + startFrame];
            }
            return output;
        }

        if (tensor.Dimensions.Length == 1) {
            var t = tensor.Dimensions[0];
            var tOut = Math.Max(0, t - startFrame);
            var output = new float[tOut];
            for (var ti = 0; ti < tOut; ti++) {
                output[ti] = tensor[ti + startFrame];
            }
            return output;
        }

        throw new InvalidOperationException($"Unexpected tensor rank for 1D extraction: {tensor.Dimensions.Length}");
    }

    private static WordList MergeWordLists(List<WordList> wordsList, double wavLength) {
        if (wordsList.Count == 0) {
            return new WordList();
        }

        if (wordsList.Count == 1) {
            return CloneWordList(wordsList[0]);
        }

        try {
            var resultWord = new WordList();
            var phonemesAll = new List<Phoneme>();
            var wordCount = wordsList[0].Count;

            for (var wIdx = 0; wIdx < wordCount; wIdx++) {
                var phonemes = new List<Phoneme>();
                var phCount = wordsList[0][wIdx].Phonemes.Count;

                for (var phIdx = 0; phIdx < phCount; phIdx++) {
                    var starts = wordsList.Select(words => words[wIdx].Phonemes[phIdx].Start).ToList();
                    var ends = wordsList.Select(words => words[wIdx].Phonemes[phIdx].End).ToList();

                    var phStart = InferMath.RemoveOutliersPerPosition(new List<List<double>> { starts })[0];
                    var phEnd = InferMath.RemoveOutliersPerPosition(new List<List<double>> { ends })[0];

                    var lastEnd = phonemesAll.Count > 0 ? phonemesAll[^1].End : 0;
                    phStart = Math.Max(phStart, lastEnd);
                    phEnd = Math.Max(phStart + 0.0001, phEnd);

                    var text = wordsList[0][wIdx].Phonemes[phIdx].Text;
                    var phoneme = new Phoneme(phStart, phEnd, text);
                    phonemes.Add(phoneme);
                    phonemesAll.Add(phoneme);
                }

                var wordText = wordsList[0][wIdx].Text;
                var word = new Word(phonemes[0].Start, phonemes[^1].End, wordText);
                foreach (var ph in phonemes) {
                    word.AddPhoneme(ph, resultWord.LogLines);
                }
                resultWord.AppendWord(word);
            }

            return resultWord;
        } catch (Exception ex) {
            Log.Warning(ex, "HubertFA: MergeWordLists failed, fallback to first pad result.");
            return CloneWordList(wordsList[0]);
        }
    }

    private static WordList CloneWordList(WordList source) {
        var cloned = new WordList();
        foreach (var word in source) {
            var w = new Word(word.Start, word.End, word.Text);
            foreach (var ph in word.Phonemes) {
                w.AddPhoneme(new Phoneme(ph.Start, ph.End, ph.Text), cloned.LogLines);
            }
            cloned.AppendWord(w);
        }
        return cloned;
    }

    private void EnsureConfigFile(string fileName) {
        var path = Path.Combine(modelFolder, fileName);
        if (!File.Exists(path)) {
            throw new FileNotFoundException($"Missing required file: {path}");
        }
    }

    private void EnsureConfigsLoaded() {
        if (vocab == null || melCfg == null) {
            throw new InvalidOperationException("Call LoadConfig() first.");
        }
    }

    private void EnsureReadyForInfer() {
        EnsureConfigsLoaded();
        if (model == null || faDecoder == null || nllDecoder == null) {
            throw new InvalidOperationException("Call InitDecoder() and LoadModel() first.");
        }
    }

    public void Dispose() {
        model?.Dispose();
        model = null;
    }
}





internal sealed class AlignmentDecoder {
    private readonly VocabConfig vocab;
    private readonly int sampleRate;
    private readonly int hopSize;
    private readonly double frameLength;

    public AlignmentDecoder(VocabConfig vocab, int sampleRate, int hopSize) {
        this.vocab = vocab;
        this.sampleRate = sampleRate;
        this.hopSize = hopSize;
        frameLength = hopSize / (double)sampleRate;
    }

    public (WordList Words, double TotalConfidence) Decode(
        float[,] phFrameLogits,
        float[] phEdgeLogits,
        double wavLength,
        List<string> phSeq,
        List<string>? wordSeq = null,
        List<int>? phIdxToWordIdx = null,
        bool ignoreSp = true
    ) {
        var phSeqId = phSeq.Select(ph => vocab.Vocab.TryGetValue(ph, out var id)
            ? id
            : throw new KeyNotFoundException($"Phoneme '{ph}' not in vocab")).ToArray();

        var phMask = Enumerable.Repeat(1e9f, vocab.VocabSize).ToArray();
        foreach (var id in phSeqId) {
            phMask[id] = 0;
        }
        phMask[0] = 0;

        if (wordSeq == null || phIdxToWordIdx == null) {
            wordSeq = new List<string>(phSeq);
            phIdxToWordIdx = Enumerable.Range(0, phSeq.Count).ToList();
        }

        var numFrames = (int)((wavLength * sampleRate + 0.5) / hopSize);
        var tRaw = phFrameLogits.GetLength(1);
        numFrames = Math.Min(numFrames, tRaw);
        numFrames = Math.Min(numFrames, phEdgeLogits.Length);

        var adjusted = new float[vocab.VocabSize, numFrames];
        for (var c = 0; c < vocab.VocabSize; c++) {
            for (var t = 0; t < numFrames; t++) {
                adjusted[c, t] = phFrameLogits[c, t] - phMask[c];
            }
        }

        var phProbLog = LogSoftmaxAxis0(adjusted);
        var phEdgePred = new float[numFrames];
        for (var t = 0; t < numFrames; t++) {
            phEdgePred[t] = Clip01(Sigmoid(phEdgeLogits[t]));
        }

        var edgeDiff = new float[numFrames];
        for (var t = 0; t < numFrames - 1; t++) {
            edgeDiff[t] = phEdgePred[t + 1] - phEdgePred[t];
        }
        edgeDiff[numFrames - 1] = 0;

        var edgeProb = new float[numFrames];
        for (var t = 0; t < numFrames; t++) {
            var prev = t == 0 ? 0 : phEdgePred[t - 1];
            edgeProb[t] = Clip01(phEdgePred[t] + prev);
        }

        var (phIdxSeq, phTimeIntPred, frameConfidence) = DecodePathLowMem(phSeqId, phProbLog, edgeProb);
        var confidenceLogs = frameConfidence.Select(x => Math.Log(x + 1e-6));
        var totalConfidence = Math.Exp(confidenceLogs.Average() / 3.0);

        var phTimePred = new double[phTimeIntPred.Count + 1];
        for (var i = 0; i < phTimeIntPred.Count; i++) {
            var frac = Clip(edgeDiff[phTimeIntPred[i]] / 2.0, -0.5, 0.5);
            phTimePred[i] = Math.Max(0, frameLength * (phTimeIntPred[i] + frac));
        }
        phTimePred[^1] = Math.Max(0, frameLength * numFrames);

        var words = new WordList();
        Word? word = null;
        var wordIdxLast = -1;

        for (var i = 0; i < phIdxSeq.Count; i++) {
            var phIdx = phIdxSeq[i];
            var phText = phSeq[phIdx];
            if (phText == "SP" && ignoreSp) {
                continue;
            }

            var start = phTimePred[i];
            var end = phTimePred[i + 1];
            var phoneme = new Phoneme(start, end, phText);

            var wordIdx = phIdxToWordIdx[phIdx];
            if (wordIdx == wordIdxLast && word != null) {
                word.AppendPhoneme(phoneme, words.LogLines);
            } else {
                word = new Word(start, end, wordSeq[wordIdx]);
                word.AddPhoneme(phoneme, words.LogLines);
                words.AppendWord(word);
                wordIdxLast = wordIdx;
            }
        }

        return (words, totalConfidence);
    }

    private (List<int> PhIdxSeq, List<int> PhTimeInt, List<float> FrameConfidence) DecodePath(
        int[] phSeqId,
        float[,] phProbLog,
        float[] edgeProb
    ) {
        var vocabSize = phProbLog.GetLength(0);
        var tCount = phProbLog.GetLength(1);
        var sCount = phSeqId.Length;

        var probLog = new float[sCount, tCount];
        for (var s = 0; s < sCount; s++) {
            var id = phSeqId[s];
            for (var t = 0; t < tCount; t++) {
                probLog[s, t] = phProbLog[id, t];
            }
        }

        var currPhMaxProbLog = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();
        var dp = new float[sCount, tCount];
        var backtrack = new int[sCount, tCount];

        for (var s = 0; s < sCount; s++) {
            for (var t = 0; t < tCount; t++) {
                dp[s, t] = float.NegativeInfinity;
                backtrack[s, t] = -1;
            }
        }

        dp[0, 0] = probLog[0, 0];
        currPhMaxProbLog[0] = probLog[0, 0];
        if (phSeqId[0] == 0 && sCount > 1) {
            dp[1, 0] = probLog[1, 0];
            currPhMaxProbLog[1] = probLog[1, 0];
        }

        ForwardPass(tCount, sCount, probLog, edgeProb, currPhMaxProbLog, dp, phSeqId, backtrack, sCount >= 2 ? 2 : 1);

        var phIdxSeq = new List<int>();
        var phTimeInt = new List<int>();
        var frameConfidence = new List<float>();

        var sIdx = sCount == 1
            ? 0
            : (dp[sCount - 2, tCount - 1] > dp[sCount - 1, tCount - 1] && phSeqId[sCount - 1] == 0 ? sCount - 2 : sCount - 1);

        for (var t = tCount - 1; t >= 0; t--) {
            if (!(backtrack[sIdx, t] >= 0 || t == 0)) {
                throw new InvalidOperationException("Backtracking failed.");
            }
            frameConfidence.Add(dp[sIdx, t]);

            if (backtrack[sIdx, t] != 0) {
                phIdxSeq.Add(sIdx);
                phTimeInt.Add(t);
                if (backtrack[sIdx, t] == 1) {
                    sIdx -= 1;
                } else if (backtrack[sIdx, t] == 2) {
                    sIdx -= 2;
                }
            }
        }

        phIdxSeq.Reverse();
        phTimeInt.Reverse();
        frameConfidence.Reverse();

        var scoreDiff = new List<float>(frameConfidence.Count);
        var prev = 0f;
        for (var i = 0; i < frameConfidence.Count; i++) {
            var diff = frameConfidence[i] - prev;
            scoreDiff.Add(MathF.Exp(diff));
            prev = frameConfidence[i];
        }

        return (phIdxSeq, phTimeInt, scoreDiff);
    }

    private static void ForwardPass(
        int tCount,
        int sCount,
        float[,] probLog,
        float[] edgeProb,
        float[] currPhMaxProbLog,
        float[,] dp,
        int[] phSeqId,
        int[,] backtrack,
        int prob3PadLen
    ) {
        var edgeProbLog = edgeProb.Select(x => MathF.Log(x + 1e-6f)).ToArray();
        var notEdgeProbLog = edgeProb.Select(x => MathF.Log(1 - x + 1e-6f)).ToArray();
        var maskReset = phSeqId.Select(x => x == 0).ToArray();
        var tsRatio = tCount / (float)sCount;

        var prob1 = new float[sCount];
        var prob2 = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();
        var prob3 = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();

        for (var t = 1; t < tCount; t++) {
            var edgeLogT = edgeProbLog[t];
            var notEdgeLogT = notEdgeProbLog[t];

            for (var s = 0; s < sCount; s++) {
                var probLogT = probLog[s, t];
                var dpPrev = dp[s, t - 1];
                prob1[s] = dpPrev + probLogT + notEdgeLogT;
            }

            for (var s = 1; s < sCount; s++) {
                prob2[s] = dp[s - 1, t - 1] + probLog[s - 1, t] + edgeLogT + currPhMaxProbLog[s - 1] * tsRatio;
            }

            for (var s = prob3PadLen; s < sCount; s++) {
                var source = s - prob3PadLen;
                var idx = Math.Clamp(s - prob3PadLen + 1, 0, sCount - 1);
                var valid = idx >= sCount - 1 || phSeqId[idx] == 0;
                if (valid) {
                    prob3[s] = dp[source, t - 1] + probLog[source, t] + edgeLogT + currPhMaxProbLog[source] * tsRatio;
                }
            }

            for (var s = 0; s < sCount; s++) {
                var max = prob1[s];
                var idx = 0;
                if (prob2[s] > max) {
                    max = prob2[s];
                    idx = 1;
                }
                if (prob3[s] > max) {
                    max = prob3[s];
                    idx = 2;
                }
                dp[s, t] = max;
                backtrack[s, t] = idx;
            }

            for (var s = 0; s < sCount; s++) {
                if (backtrack[s, t] == 0) {
                    currPhMaxProbLog[s] = MathF.Max(currPhMaxProbLog[s], probLog[s, t]);
                } else {
                    currPhMaxProbLog[s] = probLog[s, t];
                }
                if (maskReset[s]) {
                    currPhMaxProbLog[s] = 0f;
                }
            }

            for (var s = 1; s < sCount; s++) {
                prob2[s] = float.NegativeInfinity;
            }
            for (var s = prob3PadLen; s < sCount; s++) {
                prob3[s] = float.NegativeInfinity;
            }
        }
    }


    private (List<int> PhIdxSeq, List<int> PhTimeInt, List<float> FrameConfidence) DecodePathLowMem(
        int[] phSeqId,
        float[,] phProbLog,
        float[] edgeProb
    ) {
        var tCount = phProbLog.GetLength(1);
        var sCount = phSeqId.Length;

        var currPhMaxProbLog = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();
        var dp = new float[sCount, tCount];
        var backtrack = new sbyte[sCount, tCount];

        for (var s = 0; s < sCount; s++) {
            for (var t = 0; t < tCount; t++) {
                dp[s, t] = float.NegativeInfinity;
                backtrack[s, t] = -1;
            }
        }

        dp[0, 0] = phProbLog[phSeqId[0], 0];
        currPhMaxProbLog[0] = dp[0, 0];
        if (phSeqId[0] == 0 && sCount > 1) {
            dp[1, 0] = phProbLog[phSeqId[1], 0];
            currPhMaxProbLog[1] = dp[1, 0];
        }

        ForwardPassLowMem(
            tCount,
            sCount,
            phProbLog,
            edgeProb,
            currPhMaxProbLog,
            dp,
            phSeqId,
            backtrack,
            sCount >= 2 ? 2 : 1
        );

        var phIdxSeq = new List<int>();
        var phTimeInt = new List<int>();
        var frameConfidence = new List<float>();

        var sIdx = sCount == 1
            ? 0
            : (dp[sCount - 2, tCount - 1] > dp[sCount - 1, tCount - 1] && phSeqId[sCount - 1] == 0 ? sCount - 2 : sCount - 1);

        for (var t = tCount - 1; t >= 0; t--) {
            if (!(backtrack[sIdx, t] >= 0 || t == 0)) {
                throw new InvalidOperationException("Backtracking failed.");
            }

            frameConfidence.Add(dp[sIdx, t]);
            var bt = backtrack[sIdx, t];
            if (bt != 0) {
                phIdxSeq.Add(sIdx);
                phTimeInt.Add(t);
                if (bt == 1) {
                    sIdx -= 1;
                } else if (bt == 2) {
                    sIdx -= 2;
                }
            }
        }

        phIdxSeq.Reverse();
        phTimeInt.Reverse();
        frameConfidence.Reverse();

        var scoreDiff = new List<float>(frameConfidence.Count);
        var prev = 0f;
        for (var i = 0; i < frameConfidence.Count; i++) {
            var diff = frameConfidence[i] - prev;
            scoreDiff.Add(MathF.Exp(diff));
            prev = frameConfidence[i];
        }

        return (phIdxSeq, phTimeInt, scoreDiff);
    }

    private static void ForwardPassLowMem(
        int tCount,
        int sCount,
        float[,] phProbLog,
        float[] edgeProb,
        float[] currPhMaxProbLog,
        float[,] dp,
        int[] phSeqId,
        sbyte[,] backtrack,
        int prob3PadLen
    ) {
        var edgeProbLog = new float[tCount];
        var notEdgeProbLog = new float[tCount];
        for (var t = 0; t < tCount; t++) {
            edgeProbLog[t] = MathF.Log(edgeProb[t] + 1e-6f);
            notEdgeProbLog[t] = MathF.Log(1f - edgeProb[t] + 1e-6f);
        }

        var maskReset = phSeqId.Select(x => x == 0).ToArray();
        var tsRatio = tCount / (float)sCount;

        var prob1 = new float[sCount];
        var prob2 = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();
        var prob3 = Enumerable.Repeat(float.NegativeInfinity, sCount).ToArray();

        for (var t = 1; t < tCount; t++) {
            var edgeLogT = edgeProbLog[t];
            var notEdgeLogT = notEdgeProbLog[t];

            for (var s = 0; s < sCount; s++) {
                var probLogT = phProbLog[phSeqId[s], t];
                var dpPrev = dp[s, t - 1];
                prob1[s] = dpPrev + probLogT + notEdgeLogT;
            }

            for (var s = 1; s < sCount; s++) {
                var source = s - 1;
                prob2[s] = dp[source, t - 1] + phProbLog[phSeqId[source], t] + edgeLogT + currPhMaxProbLog[source] * tsRatio;
            }

            for (var s = prob3PadLen; s < sCount; s++) {
                var source = s - prob3PadLen;
                var idx = Math.Clamp(s - prob3PadLen + 1, 0, sCount - 1);
                var valid = idx >= sCount - 1 || phSeqId[idx] == 0;
                if (valid) {
                    prob3[s] = dp[source, t - 1] + phProbLog[phSeqId[source], t] + edgeLogT + currPhMaxProbLog[source] * tsRatio;
                }
            }

            for (var s = 0; s < sCount; s++) {
                var max = prob1[s];
                sbyte idx = 0;
                if (prob2[s] > max) {
                    max = prob2[s];
                    idx = 1;
                }
                if (prob3[s] > max) {
                    max = prob3[s];
                    idx = 2;
                }
                dp[s, t] = max;
                backtrack[s, t] = idx;
            }

            for (var s = 0; s < sCount; s++) {
                if (backtrack[s, t] == 0) {
                    currPhMaxProbLog[s] = MathF.Max(currPhMaxProbLog[s], phProbLog[phSeqId[s], t]);
                } else {
                    currPhMaxProbLog[s] = phProbLog[phSeqId[s], t];
                }
                if (maskReset[s]) {
                    currPhMaxProbLog[s] = 0f;
                }
            }

            for (var s = 1; s < sCount; s++) {
                prob2[s] = float.NegativeInfinity;
            }
            for (var s = prob3PadLen; s < sCount; s++) {
                prob3[s] = float.NegativeInfinity;
            }
        }
    }
    private static float[,] LogSoftmaxAxis0(float[,] x) {
        var rows = x.GetLength(0);
        var cols = x.GetLength(1);
        var y = new float[rows, cols];

        for (var t = 0; t < cols; t++) {
            var max = float.NegativeInfinity;
            for (var r = 0; r < rows; r++) {
                if (x[r, t] > max) {
                    max = x[r, t];
                }
            }

            double sumExp = 0;
            for (var r = 0; r < rows; r++) {
                sumExp += Math.Exp(x[r, t] - max);
            }
            var logSumExp = max + (float)Math.Log(sumExp);
            for (var r = 0; r < rows; r++) {
                y[r, t] = x[r, t] - logSumExp;
            }
        }

        return y;
    }

    private static float Sigmoid(float x) {
        return (float)(1.0 / (1.0 + Math.Exp(-x)));
    }

    private static float Clip01(float x) {
        if (x < 0f) {
            return 0f;
        }
        if (x > 1f) {
            return 1f;
        }
        return x;
    }

    private static double Clip(double x, double low, double high) {
        if (x < low) {
            return low;
        }
        if (x > high) {
            return high;
        }
        return x;
    }
}

internal sealed class NonLexicalDecoder {
    private readonly List<string> nonLexicalPhs;
    private readonly int sampleRate;
    private readonly int hopSize;
    private readonly double frameLength;

    public NonLexicalDecoder(List<string> classNames, int sampleRate, int hopSize) {
        nonLexicalPhs = classNames;
        this.sampleRate = sampleRate;
        this.hopSize = hopSize;
        frameLength = hopSize / (double)sampleRate;
    }

    public List<List<Word>> Decode(
        float[,] cvntLogits,
        double? wavLength = null,
        List<string>? nonLexicalPhonemes = null
    ) {
        nonLexicalPhonemes ??= new List<string>();

        var tRaw = cvntLogits.GetLength(1);
        var tCount = tRaw;
        if (wavLength.HasValue) {
            var frames = (int)((wavLength.Value * sampleRate + 0.5) / hopSize);
            tCount = Math.Min(tCount, frames);
        }

        var probs = SoftmaxClassAxis(cvntLogits, tCount);

        var nonLexicalWords = new List<List<Word>>();
        foreach (var ph in nonLexicalPhonemes) {
            var idx = nonLexicalPhs.IndexOf(ph);
            if (idx < 0) {
                throw new InvalidOperationException($"Non lexical phoneme '{ph}' not found in vocab class list.");
            }
            var classProb = new float[tCount];
            for (var t = 0; t < tCount; t++) {
                classProb[t] = probs[idx, t];
            }
            nonLexicalWords.Add(NonLexicalWords(classProb, tag: ph));
        }

        return nonLexicalWords;
    }

    private List<Word> NonLexicalWords(
        float[] prob,
        float threshold = 0.5f,
        int maxGap = 5,
        int mixFrames = 10,
        string tag = ""
    ) {
        var words = new List<Word>();
        int? start = null;
        var gapCount = 0;

        for (var i = 0; i < prob.Length; i++) {
            if (prob[i] >= threshold) {
                if (start == null) {
                    start = i;
                }
                gapCount = 0;
            } else if (start != null) {
                if (gapCount < maxGap) {
                    gapCount++;
                } else {
                    var end = i - gapCount - 1;
                    if (end > start && (end - start) >= mixFrames) {
                        var word = new Word(start.Value * frameLength, end * frameLength, tag);
                        word.AddPhoneme(new Phoneme(start.Value * frameLength, end * frameLength, tag));
                        words.Add(word);
                    }
                    start = null;
                    gapCount = 0;
                }
            }
        }

        if (start != null && (prob.Length - start.Value) >= mixFrames) {
            var word = new Word(start.Value * frameLength, (prob.Length - 1) * frameLength, tag);
            word.AddPhoneme(new Phoneme(start.Value * frameLength, (prob.Length - 1) * frameLength, tag));
            words.Add(word);
        }

        return words;
    }

    private static float[,] SoftmaxClassAxis(float[,] logits, int tCount) {
        var classes = logits.GetLength(0);
        var probs = new float[classes, tCount];

        for (var t = 0; t < tCount; t++) {
            var max = float.NegativeInfinity;
            for (var c = 0; c < classes; c++) {
                if (logits[c, t] > max) {
                    max = logits[c, t];
                }
            }

            double sum = 0;
            for (var c = 0; c < classes; c++) {
                sum += Math.Exp(logits[c, t] - max);
            }
            for (var c = 0; c < classes; c++) {
                probs[c, t] = (float)(Math.Exp(logits[c, t] - max) / sum);
            }
        }

        return probs;
    }
}


internal sealed class Phoneme {
    public double Start { get; set; }
    public double End { get; set; }
    public string Text { get; set; }

    public Phoneme(double start, double end, string text) {
        Start = Math.Max(0.0, start);
        End = end;
        Text = text;
        if (!(Start < End)) {
            throw new ArgumentException($"Phoneme invalid: text={text}, start={Start}, end={End}");
        }
    }
}

internal sealed class Word {
    private const double Eps = 1e-7;

    public double Start { get; set; }
    public double End { get; set; }
    public string Text { get; }
    public List<Phoneme> Phonemes { get; } = new();
    public double Dur => End - Start;

    public Word(double start, double end, string text, bool initPhoneme = false) {
        Start = Math.Max(0.0, start);
        End = end;
        Text = text;
        if (!(Start < End)) {
            throw new ArgumentException($"Word invalid: text={text}, start={Start}, end={End}");
        }
        if (initPhoneme) {
            Phonemes.Add(new Phoneme(Start, End, Text));
        }
    }

    public void AddPhoneme(Phoneme phoneme, List<string>? logList = null) {
        if (Math.Abs(phoneme.Start - phoneme.End) < Eps) {
            AddWarning(logList, $"{phoneme.Text} phoneme has zero duration.");
            return;
        }
        if (phoneme.Start >= Start - Eps && phoneme.End <= End + Eps) {
            Phonemes.Add(phoneme);
        } else {
            AddWarning(logList, $"{phoneme.Text} phoneme out of word boundary, skipped.");
        }
    }

    public void AppendPhoneme(Phoneme phoneme, List<string>? logList = null) {
        if (Math.Abs(phoneme.Start - phoneme.End) < Eps) {
            AddWarning(logList, $"{phoneme.Text} phoneme has zero duration.");
            return;
        }
        if (Phonemes.Count == 0) {
            if (Math.Abs(phoneme.Start - Start) < Eps) {
                Phonemes.Add(phoneme);
                End = phoneme.End;
            } else {
                AddWarning(logList, $"{phoneme.Text} phoneme start out of word boundary, skipped.");
            }
            return;
        }

        if (Math.Abs(phoneme.Start - Phonemes[^1].End) < Eps) {
            Phonemes.Add(phoneme);
            End = phoneme.End;
        } else {
            AddWarning(logList, $"{phoneme.Text} phoneme append failed due to gap/overlap.");
        }
    }

    public void MoveStart(double newStart, List<string>? logList = null) {
        if (Phonemes.Count == 0) {
            return;
        }
        if (newStart >= 0 && newStart < Phonemes[0].End) {
            Start = newStart;
            Phonemes[0].Start = newStart;
        } else {
            AddWarning(logList, $"{Text}: cannot move start to {newStart}");
        }
    }

    public void MoveEnd(double newEnd, List<string>? logList = null) {
        if (Phonemes.Count == 0) {
            return;
        }
        if (newEnd > Phonemes[^1].Start && Phonemes[^1].Start >= 0) {
            End = newEnd;
            Phonemes[^1].End = newEnd;
        } else {
            AddWarning(logList, $"{Text}: cannot move end to {newEnd}");
        }
    }

    private static void AddWarning(List<string>? logList, string message) {
        if (logList != null) {
            logList.Add($"WARNING: {message}");
        }
    }
}

internal sealed class WordList : List<Word> {
    public List<string> LogLines { get; } = new();

    public string Log() => string.Join(Environment.NewLine, LogLines);

    public void AddLog(string message) => LogLines.Add(message);

    public List<Word> OverlappingWords(Word newWord) {
        var overlapping = new List<Word>();
        foreach (var word in this) {
            if (!(newWord.End <= word.Start || newWord.Start >= word.End)) {
                overlapping.Add(word);
            }
        }
        return overlapping;
    }

    public void AppendWord(Word word) {
        if (word.Phonemes.Count == 0) {
            AddLog($"WARNING: {word.Text} has no phonemes, skipped.");
            return;
        }

        if (Count == 0) {
            base.Add(word);
            return;
        }

        if (OverlappingWords(word).Count == 0) {
            base.Add(word);
        } else {
            AddLog($"WARNING: {word.Text} overlaps existing interval, skipped.");
        }
    }

    private static List<(double Start, double End)> RemoveOverlappingIntervals((double Start, double End) rawInterval, (double Start, double End) removeInterval) {
        var (rStart, rEnd) = rawInterval;
        var (mStart, mEnd) = removeInterval;

        if (!(rStart < rEnd)) {
            throw new ArgumentException("raw interval invalid");
        }
        if (!(mStart < mEnd)) {
            throw new ArgumentException("remove interval invalid");
        }

        var overlapStart = Math.Max(rStart, mStart);
        var overlapEnd = Math.Min(rEnd, mEnd);
        if (overlapStart >= overlapEnd) {
            return new List<(double, double)> { rawInterval };
        }

        var result = new List<(double, double)>();
        if (rStart < overlapStart) {
            result.Add((rStart, overlapStart));
        }
        if (overlapEnd < rEnd) {
            result.Add((overlapEnd, rEnd));
        }
        return result;
    }

    public void AddAp(Word newWord, double minDur = 0.1) {
        try {
            if (newWord.Phonemes.Count == 0) {
                AddLog($"WARNING: {newWord.Text} has no phonemes, skipped.");
                return;
            }

            if (Count == 0) {
                AppendWord(newWord);
                return;
            }

            var overlapping = OverlappingWords(newWord);
            if (overlapping.Count == 0) {
                AppendWord(newWord);
                Sort((a, b) => a.Start.CompareTo(b.Start));
                return;
            }

            var apIntervals = new List<(double Start, double End)> { (newWord.Start, newWord.End) };
            foreach (var word in this) {
                var temp = new List<(double Start, double End)>();
                foreach (var ap in apIntervals) {
                    temp.AddRange(RemoveOverlappingIntervals(ap, (word.Start, word.End)));
                }
                apIntervals = temp;
            }

            apIntervals = apIntervals.Where(ap => ap.End - ap.Start >= minDur).ToList();
            foreach (var ap in apIntervals) {
                try {
                    AppendWord(new Word(ap.Start, ap.End, newWord.Text, true));
                } catch (Exception ex) {
                    AddLog($"ERROR: {ex.Message}");
                }
            }
            Sort((a, b) => a.Start.CompareTo(b.Start));
        } catch (Exception ex) {
            AddLog($"ERROR in AddAp: {ex.Message}");
        }
    }

    public void FillSmallGaps(double wavLength, double gapLength = 0.1) {
        try {
            if (Count == 0) {
                return;
            }
            if (this[0].Start < 0) {
                this[0].Start = 0;
            }
            if (this[0].Start > 0 && Math.Abs(this[0].Start) < gapLength && this[0].Dur > gapLength) {
                this[0].MoveStart(0, LogLines);
            }
            if (this[^1].End >= wavLength - gapLength) {
                this[^1].MoveEnd(wavLength, LogLines);
            }
            for (var i = 1; i < Count; i++) {
                if (this[i].Start - this[i - 1].End > 0 && this[i].Start - this[i - 1].End <= gapLength) {
                    this[i - 1].MoveEnd(this[i].Start, LogLines);
                }
            }
        } catch (Exception ex) {
            AddLog($"ERROR in FillSmallGaps: {ex.Message}");
        }
    }

    public void AddSp(double wavLength, string addPhone = "SP") {
        try {
            if (Count == 0) {
                return;
            }

            var wordsRes = new WordList();
            wordsRes.LogLines.AddRange(LogLines);

            if (this[0].Start > 0) {
                try {
                    wordsRes.AppendWord(new Word(0, this[0].Start, addPhone, true));
                } catch (Exception ex) {
                    wordsRes.AddLog($"ERROR: {ex.Message}");
                }
            }

            wordsRes.AppendWord(this[0]);
            for (var i = 1; i < Count; i++) {
                var word = this[i];
                if (word.Start > wordsRes[^1].End) {
                    try {
                        wordsRes.AppendWord(new Word(wordsRes[^1].End, word.Start, addPhone, true));
                    } catch (Exception ex) {
                        wordsRes.AddLog($"ERROR: {ex.Message}");
                    }
                }
                wordsRes.AppendWord(word);
            }

            if (this[^1].End < wavLength) {
                try {
                    wordsRes.AppendWord(new Word(this[^1].End, wavLength, addPhone, true));
                } catch (Exception ex) {
                    wordsRes.AddLog($"ERROR: {ex.Message}");
                }
            }

            Clear();
            AddRange(wordsRes);
            LogLines.Clear();
            LogLines.AddRange(wordsRes.LogLines);
            Check();
        } catch (Exception ex) {
            AddLog($"ERROR in AddSp: {ex.Message}");
        }
    }

    public List<string> Phonemes => this.SelectMany(w => w.Phonemes.Select(ph => ph.Text)).ToList();

    public void ClearLanguagePrefix() {
        foreach (var word in this) {
            foreach (var phoneme in word.Phonemes) {
                var parts = phoneme.Text.Split('/');
                phoneme.Text = parts[^1];
            }
        }
    }

    public bool Check() {
        if (Count == 0) {
            return true;
        }

        for (var i = 0; i < Count; i++) {
            var word = this[i];
            if (!(word.Start < word.End)) {
                AddLog($"WARNING: Word '{word.Text}' has invalid range.");
                return false;
            }
            if (word.Phonemes.Count == 0) {
                AddLog($"WARNING: Word '{word.Text}' has no phonemes.");
                return false;
            }
            if (!NearlyEqual(word.Phonemes[0].Start, word.Start)) {
                AddLog($"WARNING: Word '{word.Text}' first phoneme start mismatched.");
                return false;
            }
            if (!NearlyEqual(word.Phonemes[^1].End, word.End)) {
                AddLog($"WARNING: Word '{word.Text}' last phoneme end mismatched.");
                return false;
            }
            for (var j = 0; j < word.Phonemes.Count; j++) {
                var ph = word.Phonemes[j];
                if (!(ph.Start < ph.End)) {
                    AddLog($"WARNING: Invalid phoneme '{ph.Text}' interval.");
                    return false;
                }
                if (j < word.Phonemes.Count - 1 && !NearlyEqual(word.Phonemes[j].End, word.Phonemes[j + 1].Start)) {
                    AddLog($"WARNING: Phoneme boundary mismatch in word '{word.Text}'.");
                    return false;
                }
            }
        }

        for (var i = 0; i < Count - 1; i++) {
            if (!NearlyEqual(this[i].End, this[i + 1].Start)) {
                AddLog($"WARNING: Word boundary mismatch between '{this[i].Text}' and '{this[i + 1].Text}'.");
                return false;
            }
        }

        return true;
    }

    private static bool NearlyEqual(double a, double b) {
        return Math.Abs(a - b) < 1e-6;
    }
}





internal sealed class ModelConfig {
    [JsonPropertyName("mel_spec_config")]
    public MelSpecConfig MelSpecConfig { get; set; } = new();
}

internal sealed class MelSpecConfig {
    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; set; }

    [JsonPropertyName("hop_size")]
    public int HopSize { get; set; }
}

internal sealed class VocabConfig {
    [JsonPropertyName("non_lexical_phonemes")]
    public List<string> NonLexicalPhonemes { get; set; } = new();

    [JsonPropertyName("dictionaries")]
    public Dictionary<string, string> Dictionaries { get; set; } = new();

    [JsonPropertyName("language_prefix")]
    public bool LanguagePrefix { get; set; }

    [JsonPropertyName("vocab")]
    public Dictionary<string, int> Vocab { get; set; } = new();

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; }
}

internal readonly record struct DatasetItem(
    string WavPath,
    List<string> PhSeq,
    List<string> WordSeq,
    List<int> PhIdxToWordIdx
);

internal readonly record struct PredictionItem(
    WordList Words
);




internal static class InferMath {
    public static List<int> FindAllDuplicatePhonemes(List<List<string>> phonemeLists) {
        if (phonemeLists.Count == 1) {
            return new List<int> { 0 };
        }

        var indexDict = new Dictionary<string, List<int>>();
        for (var i = 0; i < phonemeLists.Count; i++) {
            var key = string.Join('\u001f', phonemeLists[i]);
            if (!indexDict.TryGetValue(key, out var indices)) {
                indices = new List<int>();
                indexDict[key] = indices;
            }
            indices.Add(i);
        }

        var duplicates = indexDict
            .Where(kv => kv.Value.Count > 1)
            .Select(kv => new { Key = kv.Key, Indices = kv.Value, Length = kv.Key.Count(ch => ch == '\u001f') + 1 })
            .OrderByDescending(x => x.Indices.Count)
            .ThenByDescending(x => x.Length)
            .ToList();

        if (duplicates.Count == 0) {
            throw new InvalidOperationException("No duplicate groups found across pad runs.");
        }

        return duplicates[0].Indices;
    }

    public static List<double> RemoveOutliersPerPosition(List<List<double>> dataSeries, double threshold = 1.5) {
        var processed = new List<double>(dataSeries.Count);
        foreach (var values in dataSeries) {
            if (values.Count == 0) {
                processed.Add(0.0);
                continue;
            }

            var med = Median(values);
            var mad = MedianAbsoluteDeviation(values, med);
            if (Math.Abs(mad) < 1e-12) {
                processed.Add(med);
                continue;
            }

            var retained = new List<double>();
            foreach (var x in values) {
                var z = Math.Abs((x - med) / (mad * 1.4826));
                if (z <= threshold) {
                    retained.Add(x);
                }
            }

            processed.Add(retained.Count > 0 ? retained.Average() : med);
        }

        return processed;
    }

    private static double Median(IReadOnlyList<double> values) {
        var copy = values.ToArray();
        Array.Sort(copy);
        if (copy.Length % 2 == 1) {
            return copy[copy.Length / 2];
        }
        var right = copy.Length / 2;
        return 0.5 * (copy[right - 1] + copy[right]);
    }

    private static double MedianAbsoluteDeviation(IReadOnlyList<double> values, double center) {
        var dev = values.Select(v => Math.Abs(v - center)).ToArray();
        Array.Sort(dev);
        if (dev.Length % 2 == 1) {
            return dev[dev.Length / 2];
        }
        var right = dev.Length / 2;
        return 0.5 * (dev[right - 1] + dev[right]);
    }
}






public class HubertFALyricAlignmentOptions {
    /// <summary>
    /// Minimum overlap ratio (overlap / min(note_duration, word_duration)) to be considered high confidence.
    /// Lower-confidence words are still mapped by nearest timing to preserve lyric order.
    /// </summary>
    public float MatchThreshold { get; set; } = 0.25f;
}

public class HubertFALyricAlignmentResult {
    public int SourceWordCount { get; set; }
    public int SourceNoteCount { get; set; }
    public int ResultNoteCount { get; set; }
    public int SplitCount { get; set; }
    public int SlurCount { get; set; }
    public int LowConfidenceWordCount { get; set; }
}

public static class HubertFALyricNoteAligner {
    private const string SlurLyric = "+~";
    private const int MinSplitTick = 10;

    private static readonly HashSet<string> IgnoredTokens = new(StringComparer.OrdinalIgnoreCase) {
        "SP", "AP", "SIL", "SILENCE", "PAU", "REST",
    };

    private sealed class WordTiming {
        public double StartMs { get; init; }
        public double EndMs { get; init; }
        public string Text { get; init; } = string.Empty;
    }

    private sealed class NoteTiming {
        public UNote Note { get; init; } = null!;
        public double StartMs { get; init; }
        public double EndMs { get; init; }
    }

    public static HubertFALyricAlignmentResult Align(
        UProject project,
        UWavePart wavePart,
        UVoicePart part,
        IReadOnlyList<HubertFAWordTiming> sourceWords,
        HubertFALyricAlignmentOptions? options = null) {

        if (project == null) throw new ArgumentNullException(nameof(project));
        if (wavePart == null) throw new ArgumentNullException(nameof(wavePart));
        if (part == null) throw new ArgumentNullException(nameof(part));
        if (sourceWords == null) throw new ArgumentNullException(nameof(sourceWords));

        options ??= new HubertFALyricAlignmentOptions();
        var result = new HubertFALyricAlignmentResult();

        var partStartMs = project.timeAxis.TickPosToMsPos(part.position);
        var waveStartMs = project.timeAxis.TickPosToMsPos(wavePart.position);
        var offsetMs = waveStartMs - partStartMs;

        var words = sourceWords
            .Where(word => word.EndSec > word.StartSec)
            .Select(word => new WordTiming {
                StartMs = word.StartSec * 1000.0 + offsetMs,
                EndMs = word.EndSec * 1000.0 + offsetMs,
                Text = word.Text.Trim(),
            })
            .Where(word => !string.IsNullOrWhiteSpace(word.Text))
            .Where(word => !IgnoredTokens.Contains(word.Text))
            .OrderBy(word => word.StartMs)
            .ToList();

        var notes = part.notes
            .OrderBy(note => note.position)
            .Select(note => new NoteTiming {
                Note = note,
                StartMs = project.timeAxis.TickPosToMsPos(part.position + note.position) - partStartMs,
                EndMs = project.timeAxis.TickPosToMsPos(part.position + note.End) - partStartMs,
            })
            .ToList();

        result.SourceWordCount = words.Count;
        result.SourceNoteCount = notes.Count;

        if (words.Count == 0 || notes.Count == 0) {
            foreach (var note in part.notes) note.lyric = SlurLyric;
            result.SlurCount = part.notes.Count;
            result.ResultNoteCount = part.notes.Count;
            return result;
        }

        var wordToNote = new int[words.Count];

        double maxDistanceCost = 800.0;
        double splitPenalty = 1500.0;
        double skipPenalty = 10.0;

        double[,] dp = new double[words.Count, notes.Count];
        int[,] parent = new int[words.Count, notes.Count];

        for (int w = 0; w < words.Count; ++w) {
            for (int n = 0; n < notes.Count; ++n) {
                dp[w, n] = double.MaxValue;
                parent[w, n] = -1;

                double rawDistance = Math.Abs(words[w].StartMs - notes[n].StartMs);
                double cost = Math.Min(rawDistance, maxDistanceCost);

                if (w == 0) {
                    dp[w, n] = cost + n * skipPenalty;
                } else {
                    double minPrev = double.MaxValue;
                    int bestPrevN = -1;

                    for (int prevN = 0; prevN <= n; ++prevN) {
                        if (dp[w - 1, prevN] != double.MaxValue) {
                            double transitionCost;
                            if (prevN == n) {
                                transitionCost = splitPenalty;
                            } else {
                                transitionCost = (n - prevN - 1) * skipPenalty;
                                double spanEnd = notes[n - 1].EndMs;
                                double audioEnd = words[w - 1].EndMs;
                                double spanMismatch = Math.Abs(spanEnd - audioEnd);
                                transitionCost += Math.Min(spanMismatch, maxDistanceCost) * 0.2;
                            }

                            double total = dp[w - 1, prevN] + transitionCost;
                            if (total < minPrev) {
                                minPrev = total;
                                bestPrevN = prevN;
                            }
                        }
                    }

                    if (minPrev != double.MaxValue) {
                        dp[w, n] = minPrev + cost;
                        parent[w, n] = bestPrevN;
                    }
                }
            }
        }

        double minFinalCost = double.MaxValue;
        int bestFinalN = -1;
        for (int n = 0; n < notes.Count; ++n) {
            if (dp[words.Count - 1, n] != double.MaxValue) {
                double expectedDuration = notes[notes.Count - 1].EndMs - notes[n].StartMs;
                double actualDuration = words[words.Count - 1].EndMs - words[words.Count - 1].StartMs;
                double durationMismatch = Math.Abs(expectedDuration - actualDuration);
                double totalCost = dp[words.Count - 1, n] + Math.Min(durationMismatch, maxDistanceCost) * 2.0;
                if (totalCost < minFinalCost) {
                    minFinalCost = totalCost;
                    bestFinalN = n;
                }
            }
        }

        int currN = bestFinalN;
        for (int w = words.Count - 1; w >= 0; --w) {
            wordToNote[w] = currN;
            currN = parent[w, currN];
        }

        for (int w = 0; w < words.Count; ++w) {
            var word = words[w];
            var note = notes[wordToNote[w]];
            var noteDuration = Math.Max(1e-6, note.EndMs - note.StartMs);
            var wordDuration = Math.Max(1e-6, word.EndMs - word.StartMs);
            var overlap = OverlapMs(note.StartMs, note.EndMs, word.StartMs, word.EndMs);
            var overlapScore = overlap / Math.Min(noteDuration, wordDuration);
            if (overlapScore < options.MatchThreshold) {
                result.LowConfidenceWordCount++;
            }
        }

        var groupedWordIndexes = new List<int>[notes.Count];
        for (int i = 0; i < groupedWordIndexes.Length; ++i) {
            groupedWordIndexes[i] = new List<int>();
        }
        for (int wi = 0; wi < wordToNote.Length; ++wi) {
            groupedWordIndexes[wordToNote[wi]].Add(wi);
        }

        var rebuiltNotes = new SortedSet<UNote>();
        for (int ni = 0; ni < notes.Count; ++ni) {
            var sourceNote = notes[ni].Note;
            var mappedWords = groupedWordIndexes[ni];

            if (mappedWords.Count == 0) {
                sourceNote.lyric = SlurLyric;
                rebuiltNotes.Add(sourceNote);
                result.SlurCount++;
                continue;
            }
            if (mappedWords.Count == 1) {
                sourceNote.lyric = words[mappedWords[0]].Text;
                rebuiltNotes.Add(sourceNote);
                continue;
            }

            var splitNotes = SplitNoteByWords(project, part, sourceNote, words, mappedWords, partStartMs);
            for (int si = 0; si < splitNotes.Count; ++si) {
                splitNotes[si].lyric = words[mappedWords[si]].Text;
                rebuiltNotes.Add(splitNotes[si]);
            }
            result.SplitCount += splitNotes.Count - 1;
        }

        part.notes = rebuiltNotes;
        result.ResultNoteCount = part.notes.Count;
        return result;
    }

    private static List<UNote> SplitNoteByWords(
        UProject project,
        UVoicePart part,
        UNote sourceNote,
        List<WordTiming> words,
        List<int> mappedWordIndexes,
        double partStartMs) {

        int segmentCount = mappedWordIndexes.Count;
        int dynamicMinSplitTick = Math.Max(1, Math.Min(MinSplitTick, sourceNote.duration / segmentCount));
        int originalStart = sourceNote.position;
        int originalEnd = sourceNote.End;

        var targetBoundaries = new List<int>(segmentCount - 1);
        for (int i = 1; i < segmentCount; ++i) {
            var prevWord = words[mappedWordIndexes[i - 1]];
            var currWord = words[mappedWordIndexes[i]];
            var boundaryMs = (prevWord.EndMs + currWord.StartMs) * 0.5;
            int absTick = project.timeAxis.MsPosToTickPos(partStartMs + boundaryMs);
            int relTick = absTick - part.position;
            targetBoundaries.Add(relTick);
        }

        var boundaries = NormalizeBoundaries(
            targetBoundaries,
            originalStart,
            originalEnd,
            segmentCount,
            dynamicMinSplitTick);

        if (NeedEvenSplitFallback(sourceNote.duration, boundaries, originalStart, originalEnd, segmentCount)) {
            boundaries = BuildEvenBoundaries(originalStart, originalEnd, segmentCount);
        }

        var splitNotes = new List<UNote>(segmentCount);
        int segmentStart = originalStart;
        for (int i = 0; i < segmentCount; ++i) {
            int segmentEnd = i < boundaries.Count ? boundaries[i] : originalEnd;
            if (segmentEnd < segmentStart) {
                segmentEnd = segmentStart;
            }
            var note = i == 0 ? sourceNote : sourceNote.Clone();
            note.position = segmentStart;
            note.duration = Math.Max(1, segmentEnd - segmentStart);
            splitNotes.Add(note);
            segmentStart = segmentEnd;
        }
        return splitNotes;
    }

    private static bool NeedEvenSplitFallback(
        int noteDuration,
        List<int> boundaries,
        int noteStart,
        int noteEnd,
        int segmentCount) {
        if (segmentCount <= 1) return false;
        var segmentDurations = new List<int>(segmentCount);
        int start = noteStart;
        for (int i = 0; i < segmentCount; ++i) {
            int end = i < boundaries.Count ? boundaries[i] : noteEnd;
            segmentDurations.Add(Math.Max(1, end - start));
            start = end;
        }
        int minDuration = segmentDurations.Min();
        if (segmentCount == 2) return minDuration < noteDuration * 0.2;
        int average = Math.Max(1, noteDuration / segmentCount);
        return minDuration < Math.Max(MinSplitTick, average / 4);
    }

    private static List<int> BuildEvenBoundaries(int noteStart, int noteEnd, int segmentCount) {
        var boundaries = new List<int>(Math.Max(0, segmentCount - 1));
        int total = noteEnd - noteStart;
        int prev = noteStart;
        for (int i = 1; i < segmentCount; ++i) {
            int target = noteStart + (int)Math.Round((double)total * i / segmentCount);
            int minAllowed = prev + 1;
            int maxAllowed = noteEnd - (segmentCount - i);
            if (minAllowed > maxAllowed) {
                target = prev;
            } else {
                target = Math.Clamp(target, minAllowed, maxAllowed);
            }
            boundaries.Add(target);
            prev = target;
        }
        return boundaries;
    }

    private static List<int> NormalizeBoundaries(
        List<int> rawBoundaries,
        int noteStart,
        int noteEnd,
        int segmentCount,
        int minSplitTick) {
        var normalized = new List<int>(rawBoundaries.Count);
        int prev = noteStart;
        for (int i = 0; i < rawBoundaries.Count; ++i) {
            int remainingSegments = segmentCount - (i + 1);
            int minAllowed = prev + minSplitTick;
            int maxAllowed = noteEnd - remainingSegments * minSplitTick;
            int target;
            if (minAllowed > maxAllowed) {
                target = prev;
            } else {
                target = Math.Clamp(rawBoundaries[i], minAllowed, maxAllowed);
            }
            normalized.Add(target);
            prev = target;
        }
        return normalized;
    }

    private static double OverlapMs(double aStart, double aEnd, double bStart, double bEnd) {
        var start = Math.Max(aStart, bStart);
        var end = Math.Min(aEnd, bEnd);
        return Math.Max(0, end - start);
    }
}
public sealed class HubertFALyricAlignerOptions {
    public string? ModelPath { get; set; }
    public string Language { get; set; } = "zh";
    public string G2p { get; set; } = "dictionary";
    public IG2p? G2pProvider { get; set; }
    public string NonLexicalPhonemes { get; set; } = "AP";
    public int PadTimes { get; set; } = 1;
    public int PadLength { get; set; } = 5;
    public string? DictionaryPath { get; set; }
    public HubertFALyricAlignmentOptions AlignmentOptions { get; set; } = new();
    public Action<string>? Progress { get; set; }
}

public sealed class HubertFALyricAlignerResult {
    public HubertFALyricAlignmentResult Alignment { get; init; } = new();
}

public static class HubertFALyricAligner {
    public static bool IsInstalled(string? modelPath = null) {
        return HubertFA.IsInstalled(modelPath);
    }

    public static bool TryResolveDefaultModelPath(out string path, string? modelPath = null) {
        return HubertFA.TryResolveModelPath(out path, modelPath);
    }

    public static HubertFALyricAlignerResult Align(
        UProject project,
        UWavePart wavePart,
        UVoicePart part,
        string lyrics,
        HubertFALyricAlignerOptions? options = null) {

        if (project == null) throw new ArgumentNullException(nameof(project));
        if (wavePart == null) throw new ArgumentNullException(nameof(wavePart));
        if (part == null) throw new ArgumentNullException(nameof(part));

        if (string.IsNullOrWhiteSpace(wavePart.FilePath)) {
            throw new ArgumentException("Wave path is empty.", nameof(wavePart));
        }
        if (!File.Exists(wavePart.FilePath)) {
            throw new FileNotFoundException("Wave file not found.", wavePart.FilePath);
        }
        if (!string.Equals(Path.GetExtension(wavePart.FilePath), ".wav", StringComparison.OrdinalIgnoreCase)) {
            throw new InvalidDataException($"HubertFA requires WAV input, but got '{wavePart.FilePath}'.");
        }

        options ??= new HubertFALyricAlignerOptions();
        var progress = options.Progress;

        progress?.Invoke("HubertFA 5%: Validating lyrics");
        var normalizedLyrics = NormalizeLyrics(lyrics, options.G2pProvider == null && options.Language == "zh");

        progress?.Invoke("HubertFA 10%: Preparing inference input");
        var wordTimings = HubertFA.InferWordTimings(
            wavePart.FilePath,
            normalizedLyrics,
            new HubertFAOptions {
                ModelPath = options.ModelPath,
                G2p = options.G2p,
                G2pProvider = options.G2pProvider,
                NonLexicalPhonemes = options.NonLexicalPhonemes,
                Language = options.Language,
                DictionaryPath = options.DictionaryPath,
                PadTimes = options.PadTimes,
                PadLength = options.PadLength,
            },
            progress);

        progress?.Invoke("HubertFA 95%: Applying aligned lyrics to notes");
        var alignment = HubertFALyricNoteAligner.Align(project, wavePart, part, wordTimings, options.AlignmentOptions);
        return new HubertFALyricAlignerResult {
            Alignment = alignment,
        };
    }

    private static string NormalizeLyrics(string lyrics, bool forcePinyin) {
        if (string.IsNullOrWhiteSpace(lyrics)) {
            throw new ArgumentException("Lyrics is empty.", nameof(lyrics));
        }

        var normalized = lyrics.Replace("\r", " ").Replace("\n", " ");
        var elements = new List<string>();
        var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(normalized);
        var currentWord = new System.Text.StringBuilder();

        while (enumerator.MoveNext()) {
            string s = enumerator.GetTextElement();
            if (forcePinyin && Pinyin.Pinyin.Instance.IsHanzi(s)) {
                if (currentWord.Length > 0) {
                    elements.Add(currentWord.ToString());
                    currentWord.Clear();
                }
                var pinyin = Pinyin.Pinyin.Instance.HanziToPinyin(new List<string> { s }, Pinyin.ManTone.Style.NORMAL, Pinyin.Error.Default, false, false, false).ToStrList().FirstOrDefault();
                elements.Add(pinyin ?? s);
            } else if (string.IsNullOrWhiteSpace(s) || s.Any(c => char.IsPunctuation(c) || char.IsSymbol(c))) {
                if (currentWord.Length > 0) {
                    elements.Add(currentWord.ToString());
                    currentWord.Clear();
                }
            } else {
                currentWord.Append(s);
            }
        }
        if (currentWord.Length > 0) {
            elements.Add(currentWord.ToString());
        }

        var tokens = elements
            .Select(token => token.Trim())
            .Where(token => token.Length > 0)
            .ToArray();

        if (tokens.Length == 0) {
            throw new ArgumentException("Lyrics is empty.", nameof(lyrics));
        }

        return string.Join(" ", tokens);
    }
}
