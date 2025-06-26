//
//  VimeoVideoPlayer.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 6/26/25.
//
import SwiftUI
import WebKit

struct VimeoPlayerView: UIViewRepresentable {
    let vimeoID: String
    let shouldAutoplay: Bool
    
    init(vimeoID: String, shouldAutoplay: Bool = true) {
        self.vimeoID = vimeoID
        self.shouldAutoplay = shouldAutoplay
    }
    
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.isScrollEnabled = false
        webView.navigationDelegate = context.coordinator
        return webView
    }
    
    func updateUIView(_ webView: WKWebView, context: Context) {
        let autoplayParam = shouldAutoplay ? "1" : "0"
        
        let embedHTML = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { margin: 0; padding: 0; background: black; }
                .video-container { position: relative; width: 100%; height: 100vh; }
                iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
            </style>
        </head>
        <body>
            <div class="video-container">
                <iframe src="https://player.vimeo.com/video/\(vimeoID)?autoplay=\(autoplayParam)&muted=1&badge=0&autopause=0&player_id=0&app_id=58479"
                        frameborder="0"
                        allow="autoplay; fullscreen; picture-in-picture"
                        allowfullscreen
                        title="Recovery Method Video">
                </iframe>
            </div>
        </body>
        </html>
        """
        
        webView.loadHTMLString(embedHTML, baseURL: nil)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("Vimeo video loaded successfully")
        }
        
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            print("Vimeo video failed to load: \(error.localizedDescription)")
        }
    }
}
