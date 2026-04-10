using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using OpenUtau.App.ViewModels;

namespace OpenUtau.App.Views {
    public partial class LyricAlignmentDialog : Window {
        public bool Confirmed { get; private set; }

        public LyricAlignmentDialog() {
            InitializeComponent();
        }

        void OnOkClicked(object? sender, RoutedEventArgs e) {
            Confirmed = true;
            Close();
        }

        void OnCancelClicked(object? sender, RoutedEventArgs e) {
            Close();
        }

        async void OnBrowseWavClicked(object? sender, RoutedEventArgs e) {
            if (DataContext is not LyricAlignmentViewModel vm) {
                return;
            }
            var file = await FilePicker.OpenFileAboutProject(this, "dialogs.lyricalign.selectwav", FilePicker.WAV);
            if (!string.IsNullOrWhiteSpace(file)) {
                vm.WavPath = file;
            }
        }

        protected override void OnKeyDown(KeyEventArgs e) {
            if (e.Key == Key.Escape) {
                e.Handled = true;
                Close();
            } else if (e.Key == Key.Return && e.KeyModifiers != KeyModifiers.Shift) {
                if (DataContext is LyricAlignmentViewModel vm && vm.CanRun) {
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

