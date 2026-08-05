import AVFoundation
import SwiftUI

/// Full-screen custom camera for photographing a chess board. Shows a square
/// board guide so users fill the frame with the board, plus torch and shutter
/// controls. Delivers the captured photo (or nil on cancel) via `completion`.
struct CameraCaptureView: View {
    let completion: (UIImage?) -> Void

    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.permission {
            case .authorized:
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
                BoardGuideOverlay()
                    .allowsHitTesting(false)
            case .denied:
                permissionDeniedView
            case .undetermined:
                ProgressView()
                    .tint(.white)
            }

            controls
        }
        .onAppear { camera.requestPermissionAndStart() }
        .onDisappear { camera.stop() }
    }

    // MARK: - Subviews

    private var controls: some View {
        VStack {
            HStack {
                Button {
                    completion(nil)
                } label: {
                    Image(systemName: "xmark")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(12)
                        .background(.black.opacity(0.4), in: Circle())
                }
                .accessibilityLabel("Cancel")

                Spacer()

                if camera.hasTorch {
                    Button {
                        camera.toggleTorch()
                    } label: {
                        Image(systemName: camera.isTorchOn ? "bolt.fill" : "bolt.slash.fill")
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(camera.isTorchOn ? .yellow : .white)
                            .padding(12)
                            .background(.black.opacity(0.4), in: Circle())
                    }
                    .accessibilityLabel(camera.isTorchOn ? "Turn torch off" : "Turn torch on")
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)

            Spacer()

            if camera.permission == .authorized {
                shutterButton
                    .padding(.bottom, 24)
            }
        }
    }

    private var shutterButton: some View {
        Button {
            camera.capturePhoto { image in
                completion(image)
            }
        } label: {
            ZStack {
                Circle()
                    .stroke(.white, lineWidth: 4)
                    .frame(width: 72, height: 72)
                Circle()
                    .fill(.white)
                    .frame(width: 58, height: 58)
            }
        }
        .disabled(camera.isCapturing)
        .opacity(camera.isCapturing ? 0.5 : 1)
        .accessibilityLabel("Take photo")
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Camera access is off")
                .font(.title3.bold())
                .foregroundStyle(.white)
            Text("Allow camera access in Settings to photograph your board.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

/// Semi-transparent square guide (~85% of screen width) with corner brackets
/// and a hint nudging the user to fill the frame with the board.
private struct BoardGuideOverlay: View {
    var body: some View {
        GeometryReader { geo in
            let side = geo.size.width * 0.85
            let origin = CGPoint(
                x: (geo.size.width - side) / 2,
                y: (geo.size.height - side) / 2
            )

            ZStack(alignment: .topLeading) {
                Text("Fill the frame with the board")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.black.opacity(0.5), in: Capsule())
                    .frame(width: geo.size.width)
                    .position(x: geo.size.width / 2, y: origin.y - 28)

                RoundedRectangle(cornerRadius: 12)
                    .stroke(.white.opacity(0.5), lineWidth: 2)
                    .frame(width: side, height: side)
                    .offset(x: origin.x, y: origin.y)

                CornerBrackets(cornerRadius: 12, armLength: 26)
                    .stroke(.white, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .frame(width: side, height: side)
                    .offset(x: origin.x, y: origin.y)
            }
        }
    }
}

/// Four L-shaped brackets marking the corners of a rounded rect.
private struct CornerBrackets: Shape {
    let cornerRadius: CGFloat
    let armLength: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let r = cornerRadius
        let a = armLength

        // Top left
        path.move(to: CGPoint(x: rect.minX, y: rect.minY + a))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + r))
        path.addArc(
            center: CGPoint(x: rect.minX + r, y: rect.minY + r),
            radius: r, startAngle: .degrees(180), endAngle: .degrees(270), clockwise: false
        )
        path.addLine(to: CGPoint(x: rect.minX + a, y: rect.minY))

        // Top right
        path.move(to: CGPoint(x: rect.maxX - a, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - r, y: rect.minY))
        path.addArc(
            center: CGPoint(x: rect.maxX - r, y: rect.minY + r),
            radius: r, startAngle: .degrees(270), endAngle: .degrees(0), clockwise: false
        )
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + a))

        // Bottom right
        path.move(to: CGPoint(x: rect.maxX, y: rect.maxY - a))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        path.addArc(
            center: CGPoint(x: rect.maxX - r, y: rect.maxY - r),
            radius: r, startAngle: .degrees(0), endAngle: .degrees(90), clockwise: false
        )
        path.addLine(to: CGPoint(x: rect.maxX - a, y: rect.maxY))

        // Bottom left
        path.move(to: CGPoint(x: rect.minX + a, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        path.addArc(
            center: CGPoint(x: rect.minX + r, y: rect.maxY - r),
            radius: r, startAngle: .degrees(90), endAngle: .degrees(180), clockwise: false
        )
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - a))

        return path
    }
}

/// AVCaptureVideoPreviewLayer wrapped for SwiftUI.
private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}

/// Owns the AVCaptureSession, camera permission state, torch, and photo
/// capture. All session work happens on a dedicated serial queue.
final class CameraController: NSObject, ObservableObject {

    enum Permission {
        case undetermined
        case authorized
        case denied
    }

    let session = AVCaptureSession()

    @Published private(set) var permission: Permission = .undetermined
    @Published private(set) var hasTorch = false
    @Published private(set) var isTorchOn = false
    @Published private(set) var isCapturing = false

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let photoOutput = AVCapturePhotoOutput()
    private var videoDevice: AVCaptureDevice?
    private var isConfigured = false
    /// AVCapturePhotoCaptureDelegate is not retained by AVCapturePhotoOutput;
    /// keep in-flight delegates alive here until their capture completes.
    private var inFlightDelegates: [Int64: PhotoCaptureDelegate] = [:]

    // MARK: - Lifecycle

    /// Checks (and if needed requests) camera permission, then configures and
    /// starts the session.
    func requestPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            permission = .authorized
            startSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    self.permission = granted ? .authorized : .denied
                    if granted { self.startSession() }
                }
            }
        default:
            permission = .denied
        }
    }

    func stop() {
        sessionQueue.async {
            if let device = self.videoDevice, device.hasTorch, device.torchMode == .on {
                try? device.lockForConfiguration()
                device.torchMode = .off
                device.unlockForConfiguration()
            }
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
        isTorchOn = false
    }

    private func startSession() {
        sessionQueue.async {
            self.configureIfNeeded()
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    /// Must be called on `sessionQueue`.
    private func configureIfNeeded() {
        guard !isConfigured else { return }
        isConfigured = true

        session.beginConfiguration()
        session.sessionPreset = .photo

        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
            videoDevice = device
            let torchAvailable = device.hasTorch
            DispatchQueue.main.async { self.hasTorch = torchAvailable }
        }

        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
            photoOutput.maxPhotoQualityPrioritization = .quality
        }

        session.commitConfiguration()
    }

    // MARK: - Torch

    func toggleTorch() {
        sessionQueue.async {
            guard let device = self.videoDevice, device.hasTorch else { return }
            let turnOn = device.torchMode != .on
            do {
                try device.lockForConfiguration()
                if turnOn {
                    try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
                } else {
                    device.torchMode = .off
                }
                device.unlockForConfiguration()
                DispatchQueue.main.async { self.isTorchOn = turnOn }
            } catch {
                // Torch unavailable; leave state unchanged.
            }
        }
    }

    // MARK: - Capture

    /// Captures a still photo; `completion` is called on the main queue.
    func capturePhoto(completion: @escaping (UIImage?) -> Void) {
        guard !isCapturing else { return }
        isCapturing = true

        sessionQueue.async {
            guard self.session.isRunning else {
                DispatchQueue.main.async {
                    self.isCapturing = false
                    completion(nil)
                }
                return
            }

            let settings = AVCapturePhotoSettings()
            settings.photoQualityPrioritization = .quality
            if let connection = self.photoOutput.connection(with: .video) {
                // App is iPhone portrait-only; capture upright.
                let angle: CGFloat = 90
                if connection.isVideoRotationAngleSupported(angle) {
                    connection.videoRotationAngle = angle
                }
            }

            let delegate = PhotoCaptureDelegate { [weak self] image in
                guard let self else { return }
                self.sessionQueue.async {
                    self.inFlightDelegates[settings.uniqueID] = nil
                }
                DispatchQueue.main.async {
                    self.isCapturing = false
                    completion(image)
                }
            }
            self.inFlightDelegates[settings.uniqueID] = delegate
            self.photoOutput.capturePhoto(with: settings, delegate: delegate)
        }
    }
}

/// Retained by CameraController for the duration of one capture; converts the
/// captured photo to UIImage and reports back exactly once.
private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completion: (UIImage?) -> Void
    private var image: UIImage?

    init(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if error == nil, let data = photo.fileDataRepresentation() {
            image = UIImage(data: data)
        }
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
        error: Error?
    ) {
        completion(image)
    }
}
