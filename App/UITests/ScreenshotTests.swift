import XCTest

/// Walks the screens and attaches a screenshot of each step to the test
/// result; `.github/workflows/app.yml` exports them for a person to look at.
/// The assertions only check that each step could be taken.
final class ScreenshotTests: XCTestCase {

    override func setUp() {
        continueAfterFailure = false
    }

    private func launch(_ flow: String) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["AUTOFLOW"] = flow
        app.launch()
        return app
    }

    private func snap(_ name: String) {
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }

    /// A board square by its notation; its label starts "g1, ".
    private func square(_ notation: String, in app: XCUIApplication) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", notation + ",")).firstMatch
    }

    private func tap(_ element: XCUIElement, _ what: String) {
        XCTAssertTrue(element.waitForExistence(timeout: 10), "no \(what)")
        element.tap()
    }

    /// The demo scan (`DemoScan`): White's king read as a queen on g1, the
    /// pawn on d4 as Black's.
    func testCorrections() {
        let app = launch("scanned")
        XCTAssertTrue(app.staticTexts["Check the position"].waitForExistence(timeout: 20))
        snap("corrections-1-scanned")

        // "White has no king": Show selects the least sure white queen.
        tap(app.buttons["Show"], "Show")
        snap("corrections-2-show")

        tap(app.buttons["Make it a white king"], "king suggestion")
        snap("corrections-3-king")

        tap(square("d4", in: app), "d4")
        snap("corrections-4-d4")
        tap(app.buttons["Make it a white pawn"], "pawn suggestion")

        tap(app.buttons["Turn the board a quarter"], "quarter turn")
        snap("corrections-5-quarter-turn")
    }

    func testHome() {
        let app = launch("")
        XCTAssertTrue(app.staticTexts["Positions"].waitForExistence(timeout: 20))
        snap("home")
    }
}
