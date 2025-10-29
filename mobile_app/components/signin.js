import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// secure store import
import * as SecureStore from 'expo-secure-store';

class Signin extends Component {
    constructor(props) {
        super(props);
        this.state = {
            stage: 'email',
            backend_url: '',
            email: '',
            password: ''
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.NextStage = () => {
            var current_stage = this.state.stage

            if (current_stage == 'email'){
                if (this.state.email == ''){
                    alert(url)
                }else{
                    this.setState({stage: 'password'})
                }
            }
        }

        this.Login = () => {
            var email = this.state.email
            var password = this.state.password
            var Backend_Url = this.state.backend_url

            if (email == ''){
                alert('Email is required')
            }else if (password == ''){
                alert('Password is required')
            }else{
    
                var data = new FormData() 
                data.append('email', email)
                data.append('password', password)
                
                axios.post(Backend_Url + 'signin', data)
                .then(async (res) => {
                    let result = res.data
                    console.log('***Signin result:', result, '***')
                    if(result.status == 'failed'){
                        alert('Incorrent details entered.')
                    }else if(result.status == 'successful'){
                        // store access token
                        await SecureStore.setItemAsync('token', result.access_token);
                        // store username
                        await SecureStore.setItemAsync('user_name', result.user_name);
                        
                        alert('Signin successful.')
                        this.setState({stage: 'email', email: '', password: ''})
                        this.props.navigation.popToTop();
                        this.props.navigation.navigate('Image Input')
                    }
                }).catch((error) => {
                    alert('SignIn failed, please check your details and try again. Backend Url ' + Backend_Url)
                })
            }
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
    };

    componentDidMount() {

        this.GetBackendURL()
    }


    render() {
        
        return<View style={styles.container}>
            {
                this.state.stage == 'email'
                ? <View style={{marginTop: 'auto', marginBottom: 'auto', marginLeft: 20, marginRight: 20}}>
                    <Text style={{textAlign: 'left', color: '#3f4144'}}>
                        Email
                    </Text>
                    <TextInput
                        // autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "email")}
                        placeholder=""
                        placeholderTextColor='grey'
                        value={this.state.email}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: '#00539C', borderBottomWidth: 1, marginTop: 15}}
                    />
                </View>
                : <View style={{marginTop: 'auto', marginBottom: 'auto', marginLeft: 20, marginRight: 20}}>
                    <Text style={{textAlign: 'left', color: '#3f4144'}}>
                        Password
                    </Text>
                    <TextInput
                        // autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "password")}
                        placeholder=""
                        placeholderTextColor='grey'
                        value={this.state.password}
                        secureTextEntry={true}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: '#00539C', borderBottomWidth: 1, marginTop: 15}}
                    />
                </View>
            }
            <View style={styles.bottom_bar}>
                {
                    this.state.stage == 'email'
                    ? <View>
                        <TouchableOpacity
                            onPress={() => this.NextStage()}
                            style={{width: 340, borderColor: '#00539C', borderWidth: 1, borderRadius: 10, backgroundColor: '#ffffff'}}
                        >
                            <Text style={{margin: 15, color: '#00539C', fontWeight: 'bold', fontSize: 18, textAlign: 'center'}}>
                                Next
                            </Text>
                            <Entypo name="chevron-right" size={25} color='#00539C'
                                style={{
                                    alignSelf: "center",
                                    position: "absolute",
                                    right: 5,
                                    top: 10
                                }}
                            />
                        </TouchableOpacity>
                    </View>
                    : <View>
                        <TouchableOpacity
                            onPress={() => this.Login()}
                            style={{width: 340, borderColor: '#00539C', borderWidth: 1, borderRadius: 10, backgroundColor: '#ffffff'}}
                        >
                            <Text style={{margin: 15, color: '#00539C', fontWeight: 'bold', fontSize: 18, textAlign: 'center'}}>
                                Log in
                            </Text>
                        </TouchableOpacity>
                    </View>
                }
            </View>
        </View>
    }

}

export default Signin;

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
        flexDirection: 'row',
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