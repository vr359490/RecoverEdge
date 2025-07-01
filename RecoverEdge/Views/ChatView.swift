import SwiftUI

// MARK: - Server API Models
struct ServerRequest: Codable {
    let message: String
    let conversationHistory: [ServerMessage]
}

struct ServerMessage: Codable {
    let role: String
    let content: String
}

struct ServerResponse: Codable {
    let response: String
    let error: String?
}

// MARK: - Server Communication Service
class PythonServerService {
    static let shared = PythonServerService()
    
    // Change this to your server URL - default is localhost
    private let serverURL = "http://localhost:8000"
    
    private init() {}
    
    func sendMessage(_ message: String, conversationHistory: [ChatMessage]) async throws -> String {
        guard let url = URL(string: "\(serverURL)/chat") else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30 // 30 second timeout
        
        // Convert chat messages to server format
        let serverMessages = conversationHistory.map { chatMessage in
            ServerMessage(
                role: chatMessage.isUser ? "user" : "assistant",
                content: chatMessage.text
            )
        }
        
        let requestBody = ServerRequest(
            message: message,
            conversationHistory: serverMessages
        )
        
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        
        if httpResponse.statusCode != 200 {
            throw NSError(
                domain: "PythonServer",
                code: httpResponse.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Server returned status code \(httpResponse.statusCode)"]
            )
        }
        
        let serverResponse = try JSONDecoder().decode(ServerResponse.self, from: data)
        
        if let error = serverResponse.error {
            throw NSError(
                domain: "PythonServer",
                code: 0,
                userInfo: [NSLocalizedDescriptionKey: error]
            )
        }
        
        return serverResponse.response
    }
}

// MARK: - OpenAI API Models (Keep for fallback)
struct OpenAIRequest: Codable {
    let model: String
    let messages: [OpenAIMessage]
    let temperature: Double
    let max_tokens: Int
}

struct OpenAIMessage: Codable {
    let role: String
    let content: String
}

struct OpenAIResponse: Codable {
    let choices: [Choice]
    
    struct Choice: Codable {
        let message: OpenAIMessage
    }
}

// MARK: - OpenAI Service (Keep as fallback)
class OpenAIService {
    static let shared = OpenAIService()
    
    private let apiKey = Secrets.openAIAPIKey
    private let apiURL = "https://api.openai.com/v1/chat/completions"
    
    private init() {}
    
    func sendMessage(_ message: String, conversationHistory: [ChatMessage]) async throws -> String {
        guard let url = URL(string: apiURL) else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        var messages: [OpenAIMessage] = [
            OpenAIMessage(
                role: "system",
                content: """
                You are a knowledgeable recovery and fitness assistant specializing in post-workout recovery techniques.
                You provide evidence-based advice on recovery methods including foam rolling, stretching, cold therapy,
                breathing techniques, and proper rest. Keep responses helpful, concise, and focused on practical recovery advice.
                When discussing specific techniques, provide clear instructions and explain the benefits.
                If someone mentions pain or injury, remind them to consult healthcare professionals for medical advice.
                """
            )
        ]
        
        let recentHistory = conversationHistory.suffix(10)
        for historyMessage in recentHistory {
            messages.append(OpenAIMessage(
                role: historyMessage.isUser ? "user" : "assistant",
                content: historyMessage.text
            ))
        }
        
        messages.append(OpenAIMessage(role: "user", content: message))
        
        let requestBody = OpenAIRequest(
            model: "gpt-3.5-turbo",
            messages: messages,
            temperature: 0.7,
            max_tokens: 500
        )
        
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        
        if httpResponse.statusCode != 200 {
            if let errorData = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let error = errorData["error"] as? [String: Any],
               let message = error["message"] as? String {
                throw NSError(domain: "OpenAI", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: message])
            }
            throw URLError(.badServerResponse)
        }
        
        let openAIResponse = try JSONDecoder().decode(OpenAIResponse.self, from: data)
        
        guard let firstChoice = openAIResponse.choices.first else {
            throw NSError(domain: "OpenAI", code: 0, userInfo: [NSLocalizedDescriptionKey: "No response from AI"])
        }
        
        return firstChoice.message.content
    }
}

// MARK: - Chat Models
struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp: Date
    
    init(text: String, isUser: Bool) {
        self.text = text
        self.isUser = isUser
        self.timestamp = Date()
    }
}

// MARK: - Chat View
struct ChatView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var messages: [ChatMessage] = []
    @State private var messageText: String = ""
    @State private var isLoading: Bool = false
    @State private var showingError: Bool = false
    @State private var errorMessage: String = ""
    @State private var usePythonServer: Bool = true // Toggle for server vs direct OpenAI
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Server toggle (for testing)
                ServerToggleView(usePythonServer: $usePythonServer)
                    .padding(.horizontal)
                    .padding(.top, 8)
                
                // Messages List
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            // Welcome message
                            if messages.isEmpty {
                                WelcomeMessageView(onSuggestionTap: { suggestion in
                                    messageText = suggestion
                                    sendMessage()
                                })
                                .padding(.top, 20)
                            }
                            
                            // Chat messages
                            ForEach(messages) { message in
                                MessageBubble(message: message)
                            }
                            
                            // Loading indicator
                            if isLoading {
                                LoadingMessageBubble()
                            }
                            
                            // Invisible anchor for scrolling
                            Color.clear
                                .frame(height: 1)
                                .id("bottom")
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                    }
                    .onChange(of: messages.count) { _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
                
                // Message Input
                MessageInputView(
                    messageText: $messageText,
                    isLoading: isLoading,
                    onSend: sendMessage
                )
            }
            .navigationTitle("Recovery Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .background(Color(.systemGroupedBackground))
            .alert("Error", isPresented: $showingError) {
                Button("OK") {
                    showingError = false
                }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func sendMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = ChatMessage(text: messageText, isUser: true)
        messages.append(userMessage)
        
        let currentMessage = messageText
        messageText = ""
        isLoading = true
        
        Task {
            if usePythonServer {
                await sendToPythonServer(message: currentMessage)
            } else {
                await sendToOpenAI(message: currentMessage)
            }
        }
    }
    
    private func sendToPythonServer(message: String) async {
        do {
            let response = try await PythonServerService.shared.sendMessage(message, conversationHistory: messages)
            
            await MainActor.run {
                let aiMessage = ChatMessage(text: response, isUser: false)
                withAnimation(.easeInOut) {
                    isLoading = false
                    messages.append(aiMessage)
                }
            }
        } catch {
            await MainActor.run {
                isLoading = false
                
                // Check if it's a connection error
                if (error as NSError).code == NSURLErrorCannotConnectToHost ||
                   (error as NSError).code == NSURLErrorTimedOut {
                    errorMessage = "Cannot connect to Python server. Make sure the server is running on localhost:8000"
                } else {
                    errorMessage = "Server error: \(error.localizedDescription)"
                }
                
                showingError = true
                
                // Add a fallback message
                let fallbackMessage = ChatMessage(
                    text: "I'm having trouble connecting to the server. Please make sure the Python server is running.",
                    isUser: false
                )
                messages.append(fallbackMessage)
            }
        }
    }
    
    private func sendToOpenAI(message: String) async {
        do {
            let response = try await OpenAIService.shared.sendMessage(message, conversationHistory: messages)
            
            await MainActor.run {
                let aiMessage = ChatMessage(text: response, isUser: false)
                withAnimation(.easeInOut) {
                    isLoading = false
                    messages.append(aiMessage)
                }
            }
        } catch {
            await MainActor.run {
                isLoading = false
                errorMessage = "Failed to get response: \(error.localizedDescription)"
                showingError = true
                
                let fallbackMessage = ChatMessage(
                    text: "I apologize, but I'm having trouble connecting right now. Please check your internet connection and try again.",
                    isUser: false
                )
                messages.append(fallbackMessage)
            }
        }
    }
}

// MARK: - Server Toggle View
struct ServerToggleView: View {
    @Binding var usePythonServer: Bool
    
    var body: some View {
        HStack {
            Text("Backend:")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("", selection: $usePythonServer) {
                Text("Python Server").tag(true)
                Text("Direct OpenAI").tag(false)
            }
            .pickerStyle(SegmentedPickerStyle())
            .frame(width: 200)
            
            if usePythonServer {
                Image(systemName: "circle.fill")
                    .font(.caption)
                    .foregroundColor(.orange)
                Text("localhost:8000")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Welcome Message
struct WelcomeMessageView: View {
    let onSuggestionTap: (String) -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 50))
                .foregroundColor(.brandTeal)
            
            Text("Recovery Assistant")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Ask me anything about recovery techniques, equipment, or methods. I'm here to help optimize your recovery routine!")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 20)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Try asking:")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                SuggestedQuestion(text: "How do I foam roll my IT band?") {
                    onSuggestionTap("How do I foam roll my IT band?")
                }
                SuggestedQuestion(text: "What's the best recovery for sore legs?") {
                    onSuggestionTap("What's the best recovery for sore legs?")
                }
                SuggestedQuestion(text: "How long should I stretch after a workout?") {
                    onSuggestionTap("How long should I stretch after a workout?")
                }
            }
            .padding(.top, 10)
        }
        .padding()
    }
}

struct SuggestedQuestion: View {
    let text: String
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            Text(text)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.brandTeal.opacity(0.1))
                .cornerRadius(12)
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Message Bubble
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer(minLength: 60)
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(message.text)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(
                            LinearGradient(
                                colors: [Color.brandTeal, Color.brandTealDark],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(18)
                    
                    Text(formatTime(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.trailing, 8)
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "brain.head.profile")
                            .font(.caption)
                            .foregroundColor(.brandTeal)
                            .padding(.top, 2)
                        
                        Text(message.text)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(Color(.systemGray5))
                            .foregroundColor(.primary)
                            .cornerRadius(18)
                    }
                    
                    Text(formatTime(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.leading, 28)
                }
                
                Spacer(minLength: 60)
            }
        }
    }
    
    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Loading Message
struct LoadingMessageBubble: View {
    @State private var animationPhase = 0
    
    var body: some View {
        HStack {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "brain.head.profile")
                    .font(.caption)
                    .foregroundColor(.brandTeal)
                    .padding(.top, 2)
                
                HStack(spacing: 4) {
                    ForEach(0..<3, id: \.self) { index in
                        Circle()
                            .fill(Color.secondary)
                            .frame(width: 8, height: 8)
                            .scaleEffect(animationPhase == index ? 1.2 : 0.8)
                            .animation(
                                Animation.easeInOut(duration: 0.6)
                                    .repeatForever()
                                    .delay(Double(index) * 0.2),
                                value: animationPhase
                            )
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color(.systemGray5))
                .cornerRadius(18)
            }
            
            Spacer(minLength: 60)
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.6, repeats: true) { _ in
                animationPhase = (animationPhase + 1) % 3
            }
        }
    }
}

// MARK: - Message Input
struct MessageInputView: View {
    @Binding var messageText: String
    let isLoading: Bool
    let onSend: () -> Void
    @FocusState private var isTextFieldFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            Divider()
            
            HStack(spacing: 12) {
                TextField("Ask about recovery...", text: $messageText, axis: .vertical)
                    .textFieldStyle(PlainTextFieldStyle())
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color(.systemGray6))
                    .cornerRadius(20)
                    .lineLimit(1...4)
                    .disabled(isLoading)
                    .focused($isTextFieldFocused)
                    .onSubmit {
                        if canSend {
                            onSend()
                        }
                    }
                
                Button(action: onSend) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                        .foregroundColor(canSend ? .brandTeal : .gray)
                }
                .disabled(!canSend || isLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.systemBackground))
        }
    }
    
    private var canSend: Bool {
        !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
