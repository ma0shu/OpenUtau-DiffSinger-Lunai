using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reactive.Linq;
using Avalonia.Media;
using OpenUtau.App.Controls;
using OpenUtau.Core;
using OpenUtau.Core.DiffSinger;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Voicevox;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;

namespace OpenUtau.App.ViewModels {
    public class ExpSelectorItem {
        public UExpressionDescriptor Descriptor { get; }
        public string Abbr => Descriptor.abbr;
        public string DisplayName { get; }

        public ExpSelectorItem(UExpressionDescriptor descriptor, string displayName) {
            Descriptor = descriptor;
            DisplayName = displayName;
        }

        public override string ToString() => $"{Abbr.ToUpperInvariant()}: {DisplayName}";
    }

    public class ExpSelectorViewModel : ViewModelBase, ICmdSubscriber {
        private static readonly IReadOnlyDictionary<string, string> DisplayNameResourceKeys =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) {
                [OpenUtau.Core.Format.Ustx.DYN] = "exps.display.dyn",
                [OpenUtau.Core.Format.Ustx.PITD] = "exps.display.pitd",
                [OpenUtau.Core.Format.Ustx.CLR] = "exps.display.clr",
                [OpenUtau.Core.Format.Ustx.ENG] = "exps.display.eng",
                [OpenUtau.Core.Format.Ustx.VEL] = "exps.display.vel",
                [OpenUtau.Core.Format.Ustx.VOL] = "exps.display.vol",
                [OpenUtau.Core.Format.Ustx.ATK] = "exps.display.atk",
                [OpenUtau.Core.Format.Ustx.DEC] = "exps.display.dec",
                [OpenUtau.Core.Format.Ustx.GEN] = "exps.display.gen",
                [OpenUtau.Core.Format.Ustx.GENC] = "exps.display.genc",
                [OpenUtau.Core.Format.Ustx.BRE] = "exps.display.bre",
                [OpenUtau.Core.Format.Ustx.BREC] = "exps.display.brec",
                [OpenUtau.Core.Format.Ustx.LPF] = "exps.display.lpf",
                [OpenUtau.Core.Format.Ustx.NORM] = "exps.display.norm",
                [OpenUtau.Core.Format.Ustx.MOD] = "exps.display.mod",
                [OpenUtau.Core.Format.Ustx.MODP] = "exps.display.modp",
                [OpenUtau.Core.Format.Ustx.ALT] = "exps.display.alt",
                [OpenUtau.Core.Format.Ustx.DIR] = "exps.display.dir",
                [OpenUtau.Core.Format.Ustx.SHFT] = "exps.display.shft",
                [OpenUtau.Core.Format.Ustx.SHFC] = "exps.display.shfc",
                [OpenUtau.Core.Format.Ustx.TENC] = "exps.display.tenc",
                [OpenUtau.Core.Format.Ustx.VOIC] = "exps.display.voic",
                [DiffSingerUtils.VELC] = "exps.display.velc",
                [DiffSingerUtils.ENE] = "exps.display.ene",
                [DiffSingerUtils.PEXP] = "exps.display.pexp",
                [VoicevoxUtils.VOLC] = "exps.display.volc",
                [VoicevoxUtils.SMOC] = "exps.display.smoc",
                [VoicevoxUtils.DUCM] = "exps.display.ducm",
            };
        private const string VoiceColorIntensityDisplayKey = "exps.display.voicecolorintensity";

        [Reactive] public int Index { get; set; }
        [Reactive] public int SelectedIndex { get; set; }
        [Reactive] public ExpDisMode DisplayMode { get; set; }
        [Reactive] public ExpSelectorItem? Descriptor { get; set; }
        public string Abbr {
            get {
                if (Descriptor == null) {
                    return "";
                }
                return Descriptor.Abbr;
            }
        }
        public ObservableCollection<ExpSelectorItem> Descriptors => descriptors;
        public string Header => header.Value;
        [Reactive] public IBrush TagBrush { get; set; }
        [Reactive] public IBrush Background { get; set; }

        readonly ObservableCollection<ExpSelectorItem> descriptors = new ObservableCollection<ExpSelectorItem>();
        ObservableAsPropertyHelper<string> header;
        int currentTrackNo = -1;

        public ExpSelectorViewModel() {
            DocManager.Inst.AddSubscriber(this);
            this.WhenAnyValue(x => x.DisplayMode)
                .Subscribe(_ => RefreshBrushes());
            this.WhenAnyValue(x => x.Descriptor)
                .Select(descriptor => descriptor == null ? string.Empty : descriptor.Abbr.ToUpperInvariant())
                .ToProperty(this, x => x.Header, out header);
            this.WhenAnyValue(x => x.Descriptor)
                .Subscribe(SelectionChanged);
            this.WhenAnyValue(x => x.Index, x => x.Descriptors)
                .Subscribe(tuple => {
                    SetExp(DocManager.Inst.Project.expSelectors[tuple.Item1]);
                });
            MessageBus.Current.Listen<ThemeChangedEvent>()
                .Subscribe(_ => RefreshBrushes());
            TagBrush = ThemeManager.ExpNameBrush;
            Background = ThemeManager.ExpBrush;
            OnListChange();
        }

        public bool SetExp(string abbr) {
            var descriptor = Descriptors.FirstOrDefault(d =>
                string.Equals(d.Abbr, abbr, StringComparison.OrdinalIgnoreCase));
            if (descriptor != null) {
                Descriptor = descriptor;
                SelectedIndex = Descriptors.IndexOf(descriptor);
                return true;
            }
            if (Descriptors.Count > 0) {
                var fallbackIndex = Math.Clamp(Index, 0, Descriptors.Count - 1);
                Descriptor = Descriptors[fallbackIndex];
                SelectedIndex = fallbackIndex;
            } else {
                Descriptor = null;
                SelectedIndex = -1;
            }
            return false;
        }

        public void OnSelected(bool store) {
            if (DisplayMode != ExpDisMode.Visible && Descriptor != null) {
                DocManager.Inst.ExecuteCmd(new SelectExpressionNotification(Descriptor.Abbr, Index, true));
            }
            if(store) {
                var project = DocManager.Inst.Project;
                project.expSecondary = project.expPrimary;
                project.expPrimary = Index;
            }
        }

        void SelectionChanged(ExpSelectorItem? descriptor) {
            if (descriptor != null) {
                DocManager.Inst.ExecuteCmd(new SelectExpressionNotification(descriptor.Abbr, Index, DisplayMode != ExpDisMode.Visible));
            }
            if (!string.IsNullOrEmpty(Abbr) &&
                0 <= Index &&
                Index < DocManager.Inst.Project.expSelectors.Length) {
                DocManager.Inst.Project.expSelectors[Index] = Abbr;
            }
        }

        public void OnNext(UCommand cmd, bool isUndo) {
            if (cmd is LoadProjectNotification) {
                currentTrackNo = -1;
                OnListChange();
            } else if (cmd is LoadPartNotification loadPart) {
                currentTrackNo = loadPart.part.trackNo;
                OnListChange();
            } else if (cmd is ConfigureExpressionsCommand ||
                       cmd is ValidateProjectNotification ||
                       cmd is SingersRefreshedNotification) {
                OnListChange();
            } else if (cmd is TrackCommand trackCommand) {
                if (currentTrackNo < 0 || trackCommand.track.TrackNo == currentTrackNo) {
                    OnListChange();
                }
            } else if (cmd is SelectExpressionNotification selectExpression) {
                OnSelectExp(selectExpression);
            }
        }

        private bool IsSupportedByCurrentRenderer(UExpressionDescriptor descriptor) {
            var project = DocManager.Inst.Project;
            if (!TryGetCurrentTrack(out var track) || track.RendererSettings.Renderer == null) {
                return true;
            }
            if (track.TryGetExpDescriptor(project, descriptor.abbr, out var effectiveDescriptor)) {
                return track.RendererSettings.Renderer.SupportsExpression(effectiveDescriptor);
            }
            return true;
        }

        private bool TryGetCurrentTrack(out UTrack track) {
            var project = DocManager.Inst.Project;
            if (currentTrackNo < 0 || currentTrackNo >= project.tracks.Count) {
                track = null!;
                return false;
            }
            track = project.tracks[currentTrackNo];
            return true;
        }

        private static bool TryGetLocalizedString(string key, out string value) {
            if (ThemeManager.TryGetString(key, out value) && !string.Equals(value, key, StringComparison.Ordinal)) {
                return true;
            }
            value = string.Empty;
            return false;
        }

        private static bool TryGetLocalizedName(UExpressionDescriptor descriptor, out string localizedName) {
            if (DisplayNameResourceKeys.TryGetValue(descriptor.abbr, out var key) &&
                TryGetLocalizedString(key, out localizedName)) {
                return true;
            }
            if (descriptor.abbr.StartsWith(DiffSingerUtils.VoiceColorHeader, StringComparison.OrdinalIgnoreCase) &&
                descriptor.abbr.Length > DiffSingerUtils.VoiceColorHeader.Length &&
                int.TryParse(descriptor.abbr.Substring(DiffSingerUtils.VoiceColorHeader.Length), out _)) {
                const string voiceColorPrefix = "voice color ";
                if (TryGetLocalizedString(VoiceColorIntensityDisplayKey, out var template)) {
                    var suffix = descriptor.name ?? string.Empty;
                    if (suffix.StartsWith(voiceColorPrefix, StringComparison.OrdinalIgnoreCase)) {
                        suffix = suffix.Substring(voiceColorPrefix.Length);
                    }
                    localizedName = template.Contains("{0}")
                        ? string.Format(template, suffix)
                        : $"{template} {suffix}";
                    return true;
                }
            }
            localizedName = string.Empty;
            return false;
        }

        private static string GetDisplayName(UExpressionDescriptor descriptor) {
            var originalName = string.IsNullOrWhiteSpace(descriptor.name)
                ? descriptor.abbr.ToUpperInvariant()
                : descriptor.name;
            if (!TryGetLocalizedName(descriptor, out var localizedName)) {
                return originalName;
            }
            if (!string.Equals(localizedName, originalName, StringComparison.OrdinalIgnoreCase)) {
                return $"{localizedName} ({originalName})";
            }
            return localizedName;
        }

        private void OnListChange() {
            var selectedAbbr = Descriptor?.Abbr;
            var preferredAbbr = Index >= 0 && Index < DocManager.Inst.Project.expSelectors.Length
                ? DocManager.Inst.Project.expSelectors[Index]
                : string.Empty;
            Descriptors.Clear();
            foreach (var descriptor in DocManager.Inst.Project.expressions.Values.Where(IsSupportedByCurrentRenderer)) {
                Descriptors.Add(new ExpSelectorItem(descriptor, GetDisplayName(descriptor)));
            }
            if (Descriptors.Count == 0) {
                Descriptor = null;
                SelectedIndex = -1;
                return;
            }
            if (!SetExp(preferredAbbr) && !SetExp(selectedAbbr ?? string.Empty)) {
                Descriptor = Descriptors[Math.Clamp(Index, 0, Descriptors.Count - 1)];
            }
            SelectedIndex = Descriptor == null ? -1 : Descriptors.IndexOf(Descriptor);
        }

        private void OnSelectExp(SelectExpressionNotification cmd) {
            if (Descriptors.Count == 0) {
                return;
            }
            if (cmd.SelectorIndex == Index) {
                var selectedDescriptor = Descriptors.FirstOrDefault(d =>
                    string.Equals(d.Abbr, cmd.ExpKey, StringComparison.OrdinalIgnoreCase));
                if (selectedDescriptor != null) {
                    Descriptor = selectedDescriptor;
                    SelectedIndex = Descriptors.IndexOf(selectedDescriptor);
                } else if (SelectedIndex < 0 || SelectedIndex >= Descriptors.Count) {
                    Descriptor = Descriptors[0];
                    SelectedIndex = 0;
                }
                DisplayMode = ExpDisMode.Visible;
            } else if (cmd.UpdateShadow) {
                DisplayMode = DisplayMode == ExpDisMode.Visible ? ExpDisMode.Shadow : ExpDisMode.Hidden;
            }
        }

        private void RefreshBrushes() {
            TagBrush = DisplayMode == ExpDisMode.Visible
                    ? ThemeManager.ExpActiveNameBrush
                    : DisplayMode == ExpDisMode.Shadow
                    ? ThemeManager.ExpShadowNameBrush
                    : ThemeManager.ExpNameBrush;
            Background = DisplayMode == ExpDisMode.Visible
                    ? ThemeManager.ExpActiveBrush
                    : DisplayMode == ExpDisMode.Shadow
                    ? ThemeManager.ExpShadowBrush
                    : ThemeManager.ExpBrush;
        }
    }
}
