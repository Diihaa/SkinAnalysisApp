import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// secure store import
import * as SecureStore from 'expo-secure-store';

class Signup extends Component {
    constructor(props) {
        super(props);
        this.state = {
            backend_url: '',
            firstname: '',
            lastname: '',
            email: '',
            password: '',
            allergic_to: '',
            is_firstname_focused: true,
            is_lastname_focused: false,
            is_email_focused: false,
            is_name_focused: false,
            is_password_focused: false,
            is_allergic_to_focused: false
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.HandleFocus = (state) => {
            this.setState({ [state]: true })
        }

        this.HandleBlur = (state) => {
            this.setState({ [state]: false })
        }

        this.Register = () => {
    
            var Backend_Url = this.state.backend_url
            var firstname = this.state.firstname
            var lastname = this.state.lastname
            var email = this.state.email
            var password = this.state.password
            var allergic_to = this.state.allergic_to

            if (firstname == ''){
                alert('Firstname is required')
            }else if (lastname == ''){
                alert('Lastname is required')
            }else if (email == ''){
                alert('Email is required')
            }else if (password == ''){
                alert('Password is required')
            }else{

                var data = new FormData() 
                data.append('firstname', firstname)
                data.append('lastname', lastname)
                data.append('email', email)
                data.append('password', password)
                data.append('allergic_to', allergic_to)
                
                axios.post(Backend_Url + 'signup', data)
                .then((res) => {
                    let result = res.data
                    if (result == 'Email already registered'){
                        alert('Email address already registered.')
                    }else{
                        alert('Signup successful.')
                        this.props.navigation.navigate('Sign In')
                    }
                }).catch((error) => {
                    alert('Something went wrong, check your connection and try again.')
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
            <View style={{marginTop: 'auto', marginBottom: 'auto', marginLeft: 20, marginRight: 20}}>
                <Text style={{textAlign: 'left', color: '#3f4144'}}>
                    Firstname
                </Text>
                <TextInput
                    autoFocus={true}
                    onFocus={() => this.HandleFocus('is_firstname_focused')}
                    onBlur={() => this.HandleBlur('is_firstname_focused')}
                    onChangeText={(text) => this.HandleChange(text, "firstname")}
                    placeholder=""
                    placeholderTextColor='grey'
                    value={this.state.firstname}
                    style={{alignSelf: 'center', width: '100%', borderBottomColor: this.state.is_name_focused ? '#00539C':'#3f4144', borderBottomWidth: 1, marginTop: 15}}
                />
                <Text style={{textAlign: 'left', color: '#3f4144', marginTop: 30}}>
                    Lastname
                </Text>
                <TextInput
                    // autoFocus={true}
                    onFocus={() => this.HandleFocus('is_lastname_focused')}
                    onBlur={() => this.HandleBlur('is_lastname_focused')}
                    onChangeText={(text) => this.HandleChange(text, "lastname")}
                    placeholder=""
                    placeholderTextColor='grey'
                    value={this.state.lastname}
                    style={{alignSelf: 'center', width: '100%', borderBottomColor: this.state.is_name_focused ? '#00539C':'#3f4144', borderBottomWidth: 1, marginTop: 15}}
                />
                <Text style={{textAlign: 'left', color: '#3f4144', marginTop: 30}}>
                    Email
                </Text>
                <TextInput
                    // autoFocus={true}
                    onFocus={() => this.HandleFocus('is_email_focused')}
                    onBlur={() => this.HandleBlur('is_email_focused')}
                    onChangeText={(text) => this.HandleChange(text, "email")}
                    placeholder=""
                    placeholderTextColor='grey'
                    value={this.state.email}
                    style={{alignSelf: 'center', width: '100%', borderBottomColor: this.state.is_email_focused ? '#00539C':'#3f4144', borderBottomWidth: 1, marginTop: 15}}
                />
                <Text style={{textAlign: 'left', color: '#3f4144', marginTop: 30}}>
                    Allergic to (Optional)
                </Text>
                <TextInput
                    // autoFocus={true}
                    onFocus={() => this.HandleFocus('is_allergic_to_focused')}
                    onBlur={() => this.HandleBlur('is_allergic_to_focused')}
                    onChangeText={(text) => this.HandleChange(text, "allergic_to")}
                    placeholder="Allergens (Comma separated eg Pollen, Eggs)"
                    placeholderTextColor='grey'
                    value={this.state.allergic_to}
                    style={{alignSelf: 'center', width: '100%', borderBottomColor: this.state.is_allergic_to_focused ? '#00539C':'#3f4144', borderBottomWidth: 1, marginTop: 15}}
                />
                <TextInput
                    // autoFocus={true}
                    onFocus={() => this.HandleFocus('is_password_focused')}
                    onBlur={() => this.HandleBlur('is_password_focused')}
                    onChangeText={(text) => this.HandleChange(text, "password")}
                    placeholder="Password*"
                    placeholderTextColor='grey'
                    value={this.state.password}
                    secureTextEntry={true}
                    style={{alignSelf: 'center', width: '100%', borderBottomColor: this.state.is_password_focused ? '#00539C':'#3f4144', borderBottomWidth: 1, marginTop: 50}}
                />
                <Text style={{textAlign: 'center', marginTop: 20, color: '#3f4144'}}>
                    *At least 6 characters
                </Text>
            </View>
            <View style={styles.bottom_bar}>
                <TouchableOpacity
                    onPress={() => this.Register()}
                    style={{width: 340, borderColor: '#00539C', borderWidth: 1, borderRadius: 10, backgroundColor: '#ffffff'}}
                >
                    <Text style={{margin: 15, color: '#00539C', fontWeight: 'bold', fontSize: 18, textAlign: 'center'}}>
                        Create Account
                    </Text>
                </TouchableOpacity>
            </View>
        </View>
    }

}

export default Signup;

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