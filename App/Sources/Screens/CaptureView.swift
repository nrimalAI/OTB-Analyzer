import PhotosUI
import SwiftUI

/// Entry screen: photograph a board, import from the library, or start from a
/// manually entered position.
struct CaptureView: View {
    @EnvironmentObject private var model: AppModel

    @State private var pickedItem: PhotosPickerItem?
    @State private var showCamera = false

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "checkerboard.rectangle")
                .font(.system(size: 72))
                .foregroundStyle(.secondary)

            Text("Digitize a chess position")
                .font(.title2.bold())
            Text("Take a photo of the board, then review and analyze it with Stockfish.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Spacer()

            VStack(spacing: 12) {
                if UIImagePickerController.isSourceTypeAvailable(.camera) {
                    Button {
                        showCamera = true
                    } label: {
                        Label("Take Photo", systemImage: "camera.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }

                PhotosPicker(selection: $pickedItem, matching: .images) {
                    Label("Choose from Library", systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button {
                    model.startFromScratch(startingPosition: true)
                } label: {
                    Label("Set Up Manually", systemImage: "square.grid.3x3")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 16)
        }
        .navigationTitle("OTB Analyzer")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if model.isRecognizing {
                ProgressView("Reading board…")
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    await model.startFromPhoto(image)
                }
                pickedItem = nil
            }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { image in
                showCamera = false
                if let image {
                    Task { await model.startFromPhoto(image) }
                }
            }
            .ignoresSafeArea()
        }
        .alert("Couldn't read the board", isPresented: $model.recognitionFailed) {
            Button("Set Up Manually") { model.startFromScratch(startingPosition: true) }
            Button("OK", role: .cancel) {}
        } message: {
            Text("Try a photo taken more directly over the board, or place the pieces manually.")
        }
    }
}

/// Minimal camera wrapper. A custom AVCaptureSession UI with a board-guide
/// overlay is a Phase 3 polish item.
private struct CameraPicker: UIViewControllerRepresentable {
    let completion: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(completion: completion) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let completion: (UIImage?) -> Void
        init(completion: @escaping (UIImage?) -> Void) { self.completion = completion }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            completion(info[.originalImage] as? UIImage)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            completion(nil)
        }
    }
}
