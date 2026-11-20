import AVFoundation
import CoreMotion
import PhotosUI
import SwiftUI

/// Scan screen: a dark viewfinder with corner brackets and a faint grid to
/// fill with the board, a hint that follows how the phone is held, a
/// shutter, and a way to pick an existing photo. After a photo is taken or
/// chosen it stays on screen under a scanning grid while the board is read,
/// then the app moves on to checking the position, or says why it could not
/// and what to change for the next photo.
struct ScanView: View {
    @EnvironmentObject private var model: AppModel

    @StateObject private var camera = CameraController()
    @StateObject private var tilt = TiltMonitor()
    @State private var pickedItem: PhotosPickerItem?
    /// The photo being read; nil while framing a shot.
    @State private var photo: UIImage?
    /// A picked photo is still being fetched (it can be an iCloud original).
    @State private var isLoadingPick = false
    /// The load-and-read in flight, so leaving the screen can cancel it.
    @State private var readTask: Task<Void, Never>?
    @Environment(\.scenePhase) private var scenePhase

    private static let hasCamera = AVCaptureDevice.default(for: .video) != nil
    private var hasCamera: Bool { Self.hasCamera }
    /// The last photo gave no position; its reason is on screen.
    private var hasFailed: Bool { model.scanFailure != nil }
    private var isReading: Bool { photo != nil && !hasFailed }
    /// Loading or reading a photo: no second shot or pick meanwhile.
    private var isBusy: Bool { isReading || isLoadingPick }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            viewfinder
            bottomBar
        }
        .background(Theme.darkroom.ignoresSafeArea())
        .foregroundStyle(Theme.paper)
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            photo = nil
            model.scanFailure = nil
            if hasCamera {
                camera.requestPermissionAndStart()
                if camera.permission == .authorized { tilt.start() }
            }
            #if DEBUG
            if let image = DebugPhoto.take() {
                readTask = Task { await read(image) }
            }
            #endif
        }
        .onDisappear {
            readTask?.cancel()
            camera.stop()
            tilt.stop()
            model.scanFailure = nil
        }
        .onChange(of: model.scanFailure) { _, failure in
            if let failure { AccessibilityNotification.Announcement(failure.title).post() }
        }
        .onChange(of: scenePhase) { _, phase in
            // iOS switches the torch off in the background.
            if phase == .active { camera.refreshTorchState() }
            if phase == .active, camera.permission == .authorized { tilt.start() } else { tilt.stop() }
        }
        .onChange(of: camera.permission) { _, permission in
            if permission == .authorized { tilt.start() }
        }
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            // A new photo replaces the last one's failure at once.
            model.scanFailure = nil
            photo = nil
            isLoadingPick = true
            readTask = Task {
                defer {
                    isLoadingPick = false
                    pickedItem = nil
                }
                guard let data = try? await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: data)
                else {
                    if !Task.isCancelled { model.scanFailure = .unreadablePhoto }
                    return
                }
                await read(image)
            }
        }
    }

    /// Back to the live camera after a photo that gave no position.
    private func retake() {
        model.scanFailure = nil
        photo = nil
    }

    private func setUpByHand() {
        model.scanFailure = nil
        model.path = []
        model.startFromScratch(startingPosition: true)
    }

    /// Recognition takes a fraction of a second; hold the reading state
    /// briefly so it doesn't flash by. `SIMCTL_CHILD_SCAN_HOLD=<seconds>`
    /// stretches it for screenshots.
    private static let readingHold: Duration = {
        if let text = ProcessInfo.processInfo.environment["SCAN_HOLD"], let seconds = Double(text) {
            return .seconds(seconds)
        }
        return .milliseconds(700)
    }()

    private func read(_ image: UIImage) async {
        photo = image
        // A cancelled read still running: nothing started, so do not sit on
        // "Reading the board".
        if !(await model.startFromPhoto(image, minimumDuration: Self.readingHold)) { photo = nil }
    }

    // MARK: - Bars

    private var topBar: some View {
        HStack {
            Button {
                readTask?.cancel()
                model.path = []
            } label: {
                Text("Cancel")
                    .sans(16, weight: .medium)
                    .padding(.horizontal, 12)
                    .frame(height: 44)
            }
            .buttonStyle(TextButtonStyle())
            Spacer()
        }
        .padding(.horizontal, 8)
        .frame(height: 44)
    }

    private var bottomBar: some View {
        HStack {
            PhotosPicker(selection: $pickedItem, matching: .images) {
                Text("Photos")
                    .sans(15, weight: .medium)
                    .frame(width: 76, height: 44, alignment: .leading)
            }
            .buttonStyle(TextButtonStyle())
            .disabled(isBusy)
            .accessibilityLabel("Choose a photo")

            Spacer()
            shutterButton
            Spacer()

            torchButton
                .frame(width: 76, height: 44, alignment: .trailing)
        }
        .padding(.horizontal, 26)
        .frame(height: 124)
    }

    private var shutterButton: some View {
        Button {
            camera.capturePhoto { image in
                guard let image else { return }
                readTask = Task { await read(image) }
            }
        } label: {
            Circle()
                .fill(Theme.paper)
                .padding(8)
                .overlay(Circle().strokeBorder(Theme.paper, lineWidth: 3))
                .frame(width: 76, height: 76)
        }
        .buttonStyle(TextButtonStyle())
        .disabled(camera.permission != .authorized || camera.isCapturing || isBusy || hasFailed)
        .accessibilityLabel("Take photo")
    }

    @ViewBuilder
    private var torchButton: some View {
        if camera.hasTorch, !isBusy, !hasFailed {
            Button {
                camera.toggleTorch()
            } label: {
                Text(camera.isTorchOn ? "Light on" : "Light")
                    .sans(15, weight: .medium)
                    .frame(width: 76, height: 44, alignment: .trailing)
            }
            .buttonStyle(TextButtonStyle())
            .accessibilityLabel(camera.isTorchOn ? "Turn light off" : "Turn light on")
        } else {
            Color.clear
        }
    }

    // MARK: - Viewfinder

    private var viewfinder: some View {
        GeometryReader { geo in
            let side = min(geo.size.width * 0.8, geo.size.height - 120)

            ZStack {
                feed
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()

                if hasFailed {
                    Theme.darkroom.opacity(0.55)
                }

                ZStack {
                    if isReading {
                        ScanningGrid()
                    } else if !hasFailed {
                        FramingGuide()
                    }
                    if !hasFailed {
                        CornerBrackets(armLength: 30)
                            .stroke(Theme.paper, lineWidth: 3)
                    }
                }
                .frame(width: side, height: side)
                .allowsHitTesting(false)
                .accessibilityHidden(true)

                if showsTip {
                    Text(tip)
                        .sans(14, weight: tilt.advice == .tiltDown && !isBusy ? .medium : .regular)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.6), in: Capsule())
                        .padding(.horizontal, 16)
                        .frame(maxHeight: .infinity, alignment: .bottom)
                        .padding(.bottom, 16)
                        .animation(.easeOut(duration: 0.2), value: tip)
                }

                if let failure = model.scanFailure {
                    FailureCard(
                        failure: failure,
                        retakeLabel: hasCamera && camera.permission == .authorized ? "Retake" : "Try again",
                        retake: retake,
                        setUpByHand: setUpByHand
                    )
                    .frame(maxHeight: .infinity, alignment: .bottom)
                    .padding(.bottom, 16)
                }
            }
        }
    }

    /// What fills the viewfinder: the photo being read, the live camera, or
    /// a stand-in when there is no camera to show.
    @ViewBuilder
    private var feed: some View {
        if let photo {
            Image(uiImage: photo)
                .resizable()
                .scaledToFill()
        } else if !hasCamera {
            DarkStripes()
        } else {
            switch camera.permission {
            case .authorized:
                CameraPreview(session: camera.session)
            case .denied:
                permissionDeniedView
            case .undetermined:
                DarkStripes()
            }
        }
    }

    private var showsTip: Bool {
        !hasFailed && (isBusy || !hasCamera || camera.permission == .authorized)
    }

    private var tip: String {
        if isReading { return "Reading the board\u{2026}" }
        if isLoadingPick { return "Loading the photo\u{2026}" }
        if !hasCamera { return "No camera here \u{2014} choose a photo" }
        return tilt.advice.message
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 10) {
            Text("Camera access is off")
                .serif(24)
            Text("Allow camera access in Settings to photograph your board, or choose a photo instead.")
                .sans(14)
                .foregroundStyle(Theme.paper.opacity(0.6))
                .multilineTextAlignment(.center)
            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .sans(15, weight: .medium)
                    .underline()
                    .frame(height: 44)
            }
            .buttonStyle(TextButtonStyle())
        }
        .padding(.horizontal, 40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(DarkStripes())
    }
}

/// Why the last photo gave no position and what to change, over the photo
/// itself, with the way back to the camera first.
private struct FailureCard: View {
    let failure: ScanFailure
    /// "Retake" with a camera, "Try again" where only a photo can be chosen.
    let retakeLabel: String
    let retake: () -> Void
    let setUpByHand: () -> Void

    /// VoiceOver moves to the card when it appears, not only hears its title.
    @AccessibilityFocusState private var titleFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(failure.title)
                .serif(26)
                .accessibilityAddTraits(.isHeader)
                .accessibilityFocused($titleFocused)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(failure.advice, id: \.self) { line in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\u{2022}")
                        Text(line)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .sans(15)
                    .foregroundStyle(Theme.paper.opacity(0.82))
                    .accessibilityElement(children: .combine)
                }
            }
            // Side by side, or one above the other where large text does
            // not leave room.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { buttons }
                VStack(spacing: 10) { buttons }
            }
            .padding(.top, 4)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.darkroom.opacity(0.94), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(Theme.paper.opacity(0.18), lineWidth: 1))
        .padding(.horizontal, 16)
        .onAppear { titleFocused = true }
    }

    @ViewBuilder
    private var buttons: some View {
        if failure.retakeHelps {
            Button(action: retake) {
                Text(retakeLabel)
                    .sans(16, weight: .semibold)
                    .foregroundStyle(Theme.darkroom)
                    .fixedSize()
                    .padding(.horizontal, 12)
                    .frame(maxWidth: .infinity, minHeight: 48)
                    .background(Theme.paper, in: RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(TextButtonStyle())
        }
        Button(action: setUpByHand) {
            Text("Set up by hand")
                .sans(16, weight: .medium)
                .foregroundStyle(Theme.paper)
                .fixedSize()
                .padding(.horizontal, 12)
                .frame(maxWidth: .infinity, minHeight: 48)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(Theme.paper.opacity(0.6), lineWidth: Theme.ruleWidth))
        }
        .buttonStyle(TextButtonStyle())
    }
}

/// A faint board in perspective inside the brackets while framing: the
/// board as it looks from a player's seat, the view the models read best
/// (`PerspectiveGrid`), not a square seen from straight above.
private struct FramingGuide: View {
    var body: some View {
        GeometryReader { geo in
            let grid = PerspectiveGrid(in: CGRect(origin: .zero, size: geo.size))
            Path { path in
                for line in grid.lines {
                    path.move(to: line.from)
                    path.addLine(to: line.to)
                }
            }
            .stroke(Theme.paper.opacity(0.22), style: StrokeStyle(lineWidth: 1, dash: [3, 5]))
        }
    }
}

/// Follows which way the phone points while framing, for the "tilt down"
/// hint (`FramingAdvice`). Core Motion's device motion needs no permission.
private final class TiltMonitor: ObservableObject {
    @Published private(set) var advice: FramingAdvice = .frame
    private let manager = CMMotionManager()

    func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        manager.deviceMotionUpdateInterval = 0.2
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let gravity = motion?.gravity else { return }
            let next = FramingAdvice.next(
                after: self.advice,
                cameraAngle: FramingAdvice.cameraAngle(gravityX: gravity.x, y: gravity.y, z: gravity.z))
            if next != self.advice { self.advice = next }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        advice = .frame
    }
}

#if DEBUG
/// Screenshots and UI tests: `SIMCTL_CHILD_SCAN_PHOTO=<path>` reads that
/// photo once, as if it had been picked; `blank` reads a photo with no board
/// in it. Simulator apps can read paths on the Mac.
private enum DebugPhoto {
    private static var taken = false

    static func take() -> UIImage? {
        guard !taken, let value = ProcessInfo.processInfo.environment["SCAN_PHOTO"] else { return nil }
        taken = true
        if value == "blank" {
            return UIGraphicsImageRenderer(size: CGSize(width: 900, height: 1200)).image { context in
                UIColor(red: 0.55, green: 0.5, blue: 0.45, alpha: 1).setFill()
                context.fill(CGRect(x: 0, y: 0, width: 900, height: 1200))
                UIColor(red: 0.35, green: 0.3, blue: 0.27, alpha: 1).setFill()
                context.cgContext.fillEllipse(in: CGRect(x: 200, y: 350, width: 500, height: 420))
            }
        }
        return UIImage(contentsOfFile: value)
    }
}
#endif

/// 8×8 grid with a line sweeping up and down while the board is read.
private struct ScanningGrid: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var atBottom = false

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .top) {
                Path { path in
                    let step = geo.size.width / 8
                    for i in 0..<8 {
                        let offset = CGFloat(i) * step
                        path.move(to: CGPoint(x: offset, y: 0))
                        path.addLine(to: CGPoint(x: offset, y: geo.size.height))
                        path.move(to: CGPoint(x: 0, y: offset))
                        path.addLine(to: CGPoint(x: geo.size.width, y: offset))
                    }
                }
                .stroke(Theme.paper.opacity(0.3), lineWidth: 1)

                if !reduceMotion {
                    Rectangle()
                        .fill(Theme.paper)
                        .frame(height: 2)
                        .shadow(color: Theme.paper.opacity(0.7), radius: 7)
                        .offset(y: atBottom ? geo.size.height - 2 : 0)
                        .animation(
                            .easeInOut(duration: 1).repeatForever(autoreverses: true),
                            value: atBottom)
                }
            }
            .clipped()
            .onAppear { atBottom = true }
        }
    }
}

/// Diagonal dark stripes: the viewfinder's stand-in when there is no feed.
private struct DarkStripes: View {
    var body: some View {
        Canvas { context, size in
            context.fill(
                Path(CGRect(origin: .zero, size: size)),
                with: .color(Color(red: 0x17 / 255, green: 0x17 / 255, blue: 0x16 / 255)))
            let band: CGFloat = 14
            let stripe = Color(red: 0x1E / 255, green: 0x1E / 255, blue: 0x1C / 255)
            // Bands run at 45°; step along the diagonal axis.
            let period = band * 2 * 2.squareRoot()
            var x = -size.height
            while x < size.width {
                var path = Path()
                path.move(to: CGPoint(x: x, y: size.height))
                path.addLine(to: CGPoint(x: x + size.height, y: 0))
                path.addLine(to: CGPoint(x: x + size.height + period / 2, y: 0))
                path.addLine(to: CGPoint(x: x + period / 2, y: size.height))
                path.closeSubpath()
                context.fill(path, with: .color(stripe))
                x += period
            }
        }
    }
}

/// Four square L-shaped brackets marking the corners of the board frame.
private struct CornerBrackets: Shape {
    let armLength: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let a = armLength
        // Inset by half the stroke so the brackets sit inside the frame.
        let r = rect.insetBy(dx: 1.5, dy: 1.5)

        path.move(to: CGPoint(x: r.minX, y: r.minY + a))
        path.addLine(to: CGPoint(x: r.minX, y: r.minY))
        path.addLine(to: CGPoint(x: r.minX + a, y: r.minY))

        path.move(to: CGPoint(x: r.maxX - a, y: r.minY))
        path.addLine(to: CGPoint(x: r.maxX, y: r.minY))
        path.addLine(to: CGPoint(x: r.maxX, y: r.minY + a))

        path.move(to: CGPoint(x: r.maxX, y: r.maxY - a))
        path.addLine(to: CGPoint(x: r.maxX, y: r.maxY))
        path.addLine(to: CGPoint(x: r.maxX - a, y: r.maxY))

        path.move(to: CGPoint(x: r.minX + a, y: r.maxY))
        path.addLine(to: CGPoint(x: r.minX, y: r.maxY))
        path.addLine(to: CGPoint(x: r.minX, y: r.maxY - a))

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

    /// Brings `isTorchOn` back in line with the device, which switches the
    /// torch off by itself when the app leaves the foreground.
    func refreshTorchState() {
        sessionQueue.async {
            let isOn = self.videoDevice?.isTorchActive ?? false
            DispatchQueue.main.async { self.isTorchOn = isOn }
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
