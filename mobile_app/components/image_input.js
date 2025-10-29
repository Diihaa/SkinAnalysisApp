import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
import * as ImagePicker from 'expo-image-picker';
// secure store import
import * as SecureStore from 'expo-secure-store';

class ImageInput extends Component {
    constructor(props) {
        super(props);
        this.state = {
            user_access_token: '',
            backend_url: '',
            image: null,
            image_extension: null,
            loading: false
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.PickImage = async () => {
            console.log('***Picking analysis image***')
            // No permissions request is necessary for launching the image library
            let result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.All,
                allowsEditing: false,
                aspect: [4, 3],
                quality: 1,
                base64: true,
                exif: true
            });
        
            if (!result.canceled) {

                // image base64 only
                var base64 = result.assets[0].base64;

                // get image extension
                var extension = 'png'

                this.setState({image: base64, image_extension: extension})
            }else{
                console.log('***Analysis image selection cancelled***')
            }
        };

        this.AnalyseImage = () => {
            var Backend_Url = this.state.backend_url

            if (this.state.image == null){
                alert('Image is required')
            }else{
                this.setState({loading: true})

                var data = new FormData() 
                data.append('user_access_token', this.state.user_access_token)
                data.append('image', this.state.image)
                data.append('image_extension', this.state.image_extension)
                
                axios.post(Backend_Url + 'analyzeImage', data)
                .then((res) => {
                    let result = res.data
                    console.log('***Analysis results:', result)
                    if (result == 'no human face found'){
                        this.setState({loading: false})
                        alert('No human face was found in the image you submitted.')
                        this.setState({
                            image: null,
                            image_extension: ''
                        })
                    }else if(result == 'Not authorized'){
                        alert('You are not authorized to use this app.')
                    }else{
                        this.setState({loading: false})
                        this.setState({
                            image: null,
                            image_extension: ''
                        })
                        this.props.navigation.navigate('Analysis Results', {
                            skin_type_match: result.skin_type_match,
                            skin_disorder_match: result.skin_disorder_match,
                            recommended_products_list: result.recommended_products_list,
                            recommended_routine: result.recommended_routine,
                            user_allergic_to: result.user_allergic_to,
                            done: true
                        })
                        alert('Image analysis complete.')
                        console.log(result)
                    }
                }).catch((error) => {
                    this.setState({loading: false})
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
                <View 
                    onStartShouldSetResponder={() => this.PickImage()}
                    style={{height: 170, backgroundColor: '#f2f2f2'}}
                >
                    <View style={{marginLeft: 'auto', marginRight: 'auto', marginTop: 'auto', marginBottom: 'auto', borderRadius: 50, height: 90, width: 90, backgroundColor: '#eaeaea'}}>
                        <View style={{margin: 'auto'}}>
                            <Feather name="image" size={28} color="grey" 
                                style={{marginTop: 25, marginLeft: 'auto', marginRight: 'auto'}}
                            />
                        </View>
                    </View>
                </View>
                <View style={{backgroundColor: '#f2f2f2', height: 40}}>
                    <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto'}}>
                        Select Image
                    </Text>
                </View>
                {
                    this.state.loading == true
                    ? <View>
                        <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 120, textAlign: 'center', fontSize: 18}}>
                            Analysing image, please wait...
                        </Text>
                    </View>
                    : this.state.image == null
                    ? <View>
                        <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 120, textAlign: 'center'}}>
                            Waiting for image selection
                        </Text>
                    </View>
                    : <View>
                        <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 15, textAlign: 'center'}}>
                            Your selected image:
                        </Text>
                        <Image source={{uri: `data:image/png;base64,${this.state.image}`,}} style={{backgroundColor: '#f2f2f2', width: 300, height: 400, marginLeft: 'auto', marginRight: 'auto', marginTop: 20, marginBottom: 120, resizeMode: 'cover'}}/>
                    </View>
                }
            </ScrollView>
            <View style={styles.bottom_bar}>
                <TouchableOpacity
                    key='analyse_image'
                    onPress={() => this.AnalyseImage()}
                    style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 0, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                >
                    <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#00539C', fontSize: 17}}>
                        Submit Image
                    </Text>
                </TouchableOpacity>
            </View>
        </View>
    }

}

export default ImageInput;

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
        height: 2
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