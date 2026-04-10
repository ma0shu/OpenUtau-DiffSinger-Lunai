using System.IO;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using OpenUtau.App.ViewModels;
using OpenUtau.Core;
using OpenUtau.Core.Analysis;

namespace OpenUtau.App.Views {
    public partial class TranscribeDialog : Window {
        public bool Confirmed { get; private set; }

        public TranscribeDialog() {
            InitializeComponent();
        }

        void OnOkClicked(object? sender, RoutedEventArgs e) {
            Confirmed = true;
            Close();
        }

        void OnCancelClicked(object? sender, RoutedEventArgs e) {
            Close();
        }

        void OnRmvpeRowPressed(object? sender, PointerPressedEventArgs e) {
            e.Handled = true;
            _ = HandleRmvpeRowClick();
        }

        void OnTextGridRowPressed(object? sender, PointerPressedEventArgs e) {
            e.Handled = true;
            _ = HandleHubertFaRowClick();
        }

        async System.Threading.Tasks.Task HandleHubertFaRowClick() {
            if (DataContext is not TranscribeViewModel vm) {
                return;
            }
            if (!vm.HubertFAAvailable) {
                vm.EnableTextGridLyricAlignment = false;
                HubertFALyricAligner.TryResolveDefaultModelPath(out var modelPath);
                var displayPath = string.IsNullOrWhiteSpace(modelPath)
                    ? Path.Combine(PathManager.Inst.DependencyPath, "hubertfa", "model.onnx")
                    : modelPath;
                await MessageBox.ShowError(this, new MessageCustomizableException(
                    "HubertFA not found",
                    "<translate:errors.failed.transcribe.hubertfa>",
                    new FileNotFoundException(displayPath),
                    false,
                    new[] { displayPath }));
                return;
            }
            vm.EnableTextGridLyricAlignment = !vm.EnableTextGridLyricAlignment;
        }

        async System.Threading.Tasks.Task HandleRmvpeRowClick() {
            var vm = DataContext as TranscribeViewModel;
            if (vm == null) {
                return;
            }
            if (!vm.RmvpeAvailable) {
                if (RmvpeCheck != null) {
                    RmvpeCheck.IsChecked = false;
                }
                var modelPath = RmvpeTranscriber.GetModelPath();
                await MessageBox.ShowError(this, new MessageCustomizableException(
                    "RMVPE not found",
                    "<translate:errors.failed.transcribe.rmvpe>",
                    new FileNotFoundException(modelPath),
                    false,
                    new[] { modelPath }));
                return;
            }
            var current = vm.PredictPitd;
            vm.PredictPitd = !current;
            if (RmvpeCheck != null) {
                RmvpeCheck.IsChecked = !current;
            }
        }

        protected override void OnKeyDown(KeyEventArgs e) {
            if (e.Key == Key.Escape) {
                e.Handled = true;
                Close();
            } else if (e.Key == Key.Return) {
                var vm = DataContext as TranscribeViewModel;
                if (vm?.CanRun == true) {
                    e.Handled = true;
                    Confirmed = true;
                    Close();
                }
            } else {
                base.OnKeyDown(e);
            }
        }
    }
}

