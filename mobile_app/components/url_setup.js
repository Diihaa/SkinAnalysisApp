import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
import {Backend_Url} from './backend_url'
import Logo from '../assets/icon.png'
// secure store import
import * as SecureStore from 'expo-secure-store';

class UrlSetup extends Component {
    constructor(props) {
        super(props);
        this.state = {
            backend_url: ''
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.SaveBackendUrl = async () => {
            console.log('saving backend url...')
            if (this.state.backend_url == ''){
                alert('Backend url is required')
            }else{
                const url = 'http://' + this.state.backend_url+':5000/'
                // const url = this.state.backend_url
                await SecureStore.setItemAsync('backend_url', url);
                
                alert('Backend server URL saved successfully.')
                this.props.navigation.navigate('Landing')
            }
        }
    };

    componentDidMount() {

    }


    render() {

        return<View style={styles.container}>
            <ScrollView style={styles.scroll_view}>
                <View style={{marginLeft: 20, marginRight: 20}}>
                    <Image source={Logo} style={{width: 150, height: 150, marginLeft: 'auto', marginRight: 'auto', marginTop: 150, borderRadius: 20}}/>
                    <Text style={{textAlign: 'center', color: '#00539C', marginTop: 60, fontWeight: 'bold', fontSize: 18}}>
                        Backend Url setup
                    </Text>
                    <TextInput
                        autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "backend_url")}
                        placeholder="Backend Server Url e.g 192.168.1.1"
                        placeholderTextColor='grey'
                        value={this.state.backend_url}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: '#00539C', borderBottomWidth: 1, marginTop: 40}}
                    />
                </View>
            </ScrollView>
            <View style={styles.bottom_bar}>
                <TouchableOpacity
                    key='save_url'
                    onPress={() => this.SaveBackendUrl()}
                    style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 0, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                >
                    <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#00539C', fontSize: 17}}>
                        Next {'>'}
                    </Text>
                </TouchableOpacity>
            </View>
        </View>
    }

}

export default UrlSetup;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    scroll_view: {
        height: 2
    },
    bottom_bar: {
        // borderTopWidth: 1,
        // borderTopColor: 'gold',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 60,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});