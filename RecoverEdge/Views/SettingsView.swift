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
