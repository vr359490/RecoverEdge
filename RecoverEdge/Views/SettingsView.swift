//
//  SettingsView.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 7/3/25.
//

import SwiftUI
import WebKit

struct SettingsRow: View {
    let iconName: String
    let title: String
    
    var body: some View {
        HStack {
            Image(systemName: iconName)
                .foregroundColor(.gray)
                .frame(width: 24, height: 24)
            
            Text(title)
                .foregroundColor(.primary)
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
        }
        .padding(.vertical, 12)
        .padding(.horizontal)
    }
}

struct SettingsListView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                
                NavigationLink(destination: AvailableEquipmentView()) {
                    SettingsRow(iconName: "dumbbell", title: "Available Equipment")
                }

                Divider()

                NavigationLink(destination: RecoveryLibraryView().environmentObject(dataStore)) {
                    SettingsRow(iconName: "books.vertical", title: "Recovery Library")
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(UIColor.secondarySystemBackground))
            )
            .padding()
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Recovery Library View
struct RecoveryLibraryView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var showingResearch: RecoveryMethod?
    
    var body: some View {
        NavigationView {
            List {
                Section("All Recovery Methods") {
                    ForEach(dataStore.recoveryMethods) { method in
                        MethodListRow(
                            method: method,
                            showResearchAction: { showingResearch = method }
                        )
                    }
                }
            }
            .navigationTitle("Recovery Library")
        }
        .sheet(item: $showingResearch) { method in
            ResearchView(method: method)
        }
    }
}
