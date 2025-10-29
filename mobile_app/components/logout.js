import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// secure store import
import * as SecureStore from 'expo-secure-store';

class Logout extends Component {
    constructor(props) {
        super(props);
        this.state = {
            user_access_token: '',
            user_name: ''
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.Logout = async () => {
            // delete token key
            await SecureStore.deleteItemAsync('token')
            // delete username key
            await SecureStore.deleteItemAsync('user_name')

            alert('Logout successful')
            this.props.navigation.navigate('LandingRoutes')
        }

        this.GetBackendURL = async () => {
            var should_reload = false

            // get backend url
            let backend_url = await SecureStore.getItemAsync('backend_url');
            if (backend_url){
              this.setState({backend_url: backend_url})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetBackendURL()}, 1000) }
        }

        this.GetUserDetails = async () => {
            var should_reload = false

            // get access token
            let token = await SecureStore.getItemAsync('token');
            if (token){
              this.setState({user_access_token: token})
            }else{ should_reload = true }

            // get username
            let user_name = await SecureStore.getItemAsync('user_name');
            if (user_name){
              this.setState({user_name: user_name})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetUserDetails()}, 1000) }
        }
    };

    componentDidMount() {
        // get backend url
        this.GetBackendURL()
        // get user details
        this.GetUserDetails()
    }


    render() {

        return<View style={styles.container}>
            <ScrollView style={styles.scroll_view}>
                <Text style={{color: '#00539C', fontWeight: 'bold', fontSize: 18, marginTop: 180, textAlign: 'center'}}>
                    {this.state.user_name}
                </Text>
                <TouchableOpacity
                    key='logout'
                    onPress={() => this.Logout()}
                    style={{backgroundColor: '#00539C', marginLeft: 'auto', marginRight: 'auto', marginTop: 100, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                >
                    <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#FFFFFF', fontSize: 17}}>
                        Logout
                    </Text>
                </TouchableOpacity>
            </ScrollView>
        </View>
    }

}

export default Logout;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    header: {
        height: 40
    },
    search_bar: {
        // borderBottomWidth: 0.1,
        // borderBottomColor: '#eaeaea',
        flexDirection: 'row',
        backgroundColor: '#f2f2f2',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 50,
        marginTop: 50,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    top_bar: {
        borderBottomWidth: 0.1,
        borderBottomColor: '#eaeaea',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 100,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    scroll_view: {
        height: 2,
        marginLeft: 10,
        marginRight: 10
    },
    bottom_bar: {
        flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 70,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});